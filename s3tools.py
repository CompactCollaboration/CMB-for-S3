import numpy as np
from math import gcd

# # --- OPTIMIZATION: Hardcoded constant to avoid importing Scipy globally ---
SPEED_OF_LIGHT_KM_S = 299792.458
# # ------------------------------------------------------------------------

def get_default_parameters():
    params = {
        'OmegaK' : -1e-3,
        'lmax' : 20,
        'compute_kmax_internally': False,
        'compute_kmax_tol': 0.005,
        'kmax' : 1e-3,
        'accboost' : 2,
        'onlyTT' : False,

        # cosmological parameters
        'ombh2' : 0.022,
        'omch2' : 0.122,
        'mnu' : 0.06,
        'H0' : 67.5,
        'tau' : 0.06,
        'As' : 2e-9,
        'ns' : 0.965,
        
        # lens space parameters
        'p' : 5,
        'q' : 2,
        'obs_ang' : [0.,0.,0.],

        # runtime parameters
        'num_workers': None,
        'batchsize': None,
        'use_tqdm': False,
        'verbose':True,
    }
    return params

def R2omk(R,H0=67.5):
    """Calculates OmegaK given curvature radius."""
    c_H0 = SPEED_OF_LIGHT_KM_S / H0
    return -((c_H0 / R) ** 2)

def omk2R(omk, H0=67.5):
    """Calculates curvature radius given OmegaK."""
    c_H0 = SPEED_OF_LIGHT_KM_S / H0
    return c_H0 / np.sqrt(np.abs(omk))

def omk2K(omk, H0=67.5):
    """Calculates curvature given curvature OmegaK."""
    c_H0 = SPEED_OF_LIGHT_KM_S / H0
    return np.abs(omk) / (c_H0**2)

def n2k(n,Rc):
        return (n + 1) / Rc

def k2n(k,Rc):
    return int(np.round(Rc * k)) - 1

def lmindex(n, lmin=2):
    """Returns (l,m) pair for a given n."""
    l = int(np.floor(np.sqrt(n + lmin**2)))
    m = n - l * (l + 1) + lmin**2
    return np.array([l, m])

def nindex(l, m, lmin=2):
    """Returns n for a given (l,m) pair."""
    return l * (l + 1) - lmin**2 + m

def get_m_ordering_indices(lmax, lmin=2, positive_m_only=False):
    """Returns the indices required to reorder an (l,m) array into (m,l) ordering."""
    num_lm = lmax * (lmax + 2) - (lmin**2 - 1)
    
    lm_tuples = []
    for i in range(num_lm):
        l, m = lmindex(i, lmin=lmin)
        if positive_m_only and m < 0:
            continue
        lm_tuples.append((l, m, i))
        
    lm_tuples.sort(key=lambda x: (x[1], x[0]))
    
    return np.array([x[2] for x in lm_tuples])

def find_mLmR_pairs(n, p, q, output_as_index=False):
    """Find all (mL,mR) indices given n and a lens space specified by p and q."""
    a, b = q + 1, q - 1
    mod_val = 2 * p

    ML = np.arange(-n, n + 1, 2)
    g = gcd(b, mod_val)
    targets = (a * ML) % mod_val
    valid_mask = targets % g == 0

    ML = ML[valid_mask]
    targets = targets[valid_mask]

    if len(ML) == 0:
        return []

    b_red = b // g
    mod_red = mod_val // g
    targets_red = targets // g

    inv_b = pow(b_red, -1, mod_red)
    MR_base = (targets_red * inv_b) % mod_red

    results = []
    k_min = np.ceil((-n - MR_base) / mod_red).astype(int)
    k_max = np.floor((n - MR_base) / mod_red).astype(int)
    max_k_count = np.max(k_max - k_min) + 1

    for k_offset in range(max_k_count):
        k = k_min + k_offset
        active = k <= k_max
        if not np.any(active):
            break

        MR_vals = MR_base[active] + k[active] * mod_red
        ML_vals = ML[active]

        parity_mask = MR_vals % 2 == n % 2

        if np.any(parity_mask):
            valid_ML = ML_vals[parity_mask]
            valid_MR = MR_vals[parity_mask]

            if output_as_index:
                chunk = np.column_stack((valid_ML/2 + n/2, valid_MR/2 + n/2)).astype(int)
            else:
                chunk = np.column_stack((valid_ML / 2, valid_MR / 2))

            results.append(chunk)

    return np.vstack(results) if results else np.array([])

def get_available_cores():
    """Get available CPU cores for matrix calculations."""
    import os
    
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus is not None:
        return int(slurm_cpus)
        
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass 

    return os.cpu_count() or 4