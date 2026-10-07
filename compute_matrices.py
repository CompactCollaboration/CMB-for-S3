"""
This file gives an example of how to calculate multiple lens spaces on a cluster.
This file essentially loops over the specified lens spaces and saves the normalized CMB matrix and KL divergence as .npz files, 
which can then later be loaded for other calculations or to make plots.
This file can easily be adapted to suit your needs.
"""

import numpy as np
import time
from s3topologies import LensSpace
from cosmotools import get_omk_from_clone_distance
import os
import argparse
import json
import logging
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from s3tools import get_available_cores, get_default_parameters


# Set this to your release tag or Git commit when publishing.
CODE_VERSION = "1.0.0"


def json_default(value):
    """Convert NumPy values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def installed_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"

from zipfile import BadZipFile


def saved_results_match(
    matrix_file, kl_file, requested_params, software_versions
):
    if not (matrix_file.is_file() and kl_file.is_file()):
        return False

    # Normalize NumPy arrays/scalars to ordinary JSON values.
    expected_params = json.loads(json.dumps(
        requested_params,
        default=json_default,
        allow_nan=False,
    ))

    try:
        for filename, array_name in (
            (matrix_file, "norm_C_matrix"),
            (kl_file, "KL"),
        ):
            with np.load(filename, allow_pickle=False) as saved:
                if array_name not in saved.files:
                    return False

                metadata = json.loads(saved["metadata_json"].item())

                if not isinstance(metadata, dict):
                    return False
                if metadata.get("requested_parameters") != expected_params:
                    return False
                if metadata.get("software_versions") != software_versions:
                    return False

    except (
        OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile
    ) as error:
        logging.warning(
            "Cannot check saved results; recomputing: %s", error
        )
        return False

    return True

software_versions = {
    "code": CODE_VERSION,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scipy": installed_version("scipy"),
    "camb": installed_version("camb"),
}

def main():

    # update these for your specific run:
    p_q_vals = np.array([[25,2],[25,7],[25,11]])
    obs_ang_vals = np.array([[0,0,0],[0.3,0.1,0.5],[np.pi/4,0.8,2.0]])
    dnc_vals = np.linspace(0.9,1.4,10)

    parser = argparse.ArgumentParser(
    description="Compute lens-space CMB matrices and KL divergences."
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('matrix_outputs'),
        help="Directory for saved results.",
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help="Worker processes; defaults to half the available CPUs.",
    )
    args = parser.parse_args()

    available_cpus = get_available_cores()
    WORKERS = (
        max(1, available_cpus // 2)
        if args.workers is None
        else args.workers
    )

    if not 1 <= WORKERS <= available_cpus:
        parser.error(
            f"--workers must be between 1 and {available_cpus}."
        )

    BATCH_SIZE = 24

    path = args.output_dir
    kl_dir = path / 'KL_results'

    path.mkdir(parents=True, exist_ok=True)
    kl_dir.mkdir(parents=True, exist_ok=True)

    lmax = 30

    count = 0
    for p,q in p_q_vals:
        for obs_ang in obs_ang_vals:
            for dnc in dnc_vals:
                start_time = time.time()

                count +=1
                omk = get_omk_from_clone_distance(p=p, ratio=dnc, q=q, chi0=obs_ang[0])

                print(f"\nIteration {count}/{len(p_q_vals)*len(obs_ang_vals)*len(dnc_vals)}",flush=True)

                lens_name = f'Lens_{p}_{q}_lmax-{lmax}_chi0-{np.round(obs_ang[0],4)}_xip0-{np.round(obs_ang[1],4)}_xim0-{np.round(obs_ang[2],4)}'
                matrix_file = path / (lens_name + f'_omk-{np.round(abs(omk), 6)}.npz')
                kl_file = kl_dir / ('KL_' + lens_name + f'_omk-{np.round(abs(omk), 6)}.npz')

                params = {
                    'OmegaK' : omk,
                    'lmax' : lmax,
                    'compute_kmax_internally': True,
                    'p' : int(p),
                    'q' : int(q),
                    'obs_ang' : obs_ang,
                    'onlyTT' : False,
                    'verbose': False,
                    'use_tqdm':False,
                    'num_workers' :WORKERS,
                    'batchsize': BATCH_SIZE
                    }

                print(f"Initializing LensSpace L({params['p']},{params['q']}) with OmegaK={params['OmegaK']}, chi0={params['obs_ang'][0]}...",flush=True)
                # Capture all requested settings, including defaults.
                requested_params = get_default_parameters()
                requested_params.update(params)

                if saved_results_match(matrix_file, kl_file, requested_params, software_versions):
                    print("Saved matrix and KL match the requested settings. Skipping.")
                    continue

                lens = LensSpace(requested_params)

                print(f"Computed nmax: {lens.nmax}",flush=True)

                lens.compute_Clmlpmp()

                KL = lens.compute_KL()

                effective_params = lens.params.copy()
                effective_params['kmax'] = float(lens.kmax)

                metadata = {
                    'requested_parameters': requested_params,
                    'effective_parameters': effective_params,
                    'nmax': int(lens.nmax),
                    'threads_per_worker': max(1, available_cpus // WORKERS),
                    'kl_order': ['forward', 'backward'],
                    'software_versions': software_versions,
                }

                metadata_json = json.dumps(
                    metadata,
                    default=json_default,
                    allow_nan=False,
                    indent=2,
                )

                np.savez(
                    matrix_file,
                    norm_C_matrix=lens.norm_C_matrix,
                    metadata_json=np.asarray(metadata_json),
                )
                np.savez(kl_file,
                         KL=KL,
                         metadata_json=np.asarray(metadata_json),
                         )
                total_time = time.time() - start_time
                print(f"\nFinished in {total_time//3600} hrs {(total_time%3600)//60} mins {total_time % 60} secs.",flush=True)

                del lens
                time.sleep(2)

if __name__ == "__main__":
    import multiprocessing
    import sys

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        print(">>> Warning: Could not set start method to 'spawn'")

    try:
        main()
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)

