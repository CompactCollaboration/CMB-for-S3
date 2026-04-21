import os

import time
import concurrent.futures
import numpy as np
from math import pi, gcd, sqrt
from warnings import warn
from multiprocessing import shared_memory
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from s3tools import *
from wigner_d_jacobi import get_wigner_d_matrix_optimized
from fast_wigner import init_wigner_tables, free_wigner_tables, compute_wig3j_flat

class SphericalTopology():

    def __init__(self,params : dict = None) -> None:
        self.params = get_default_parameters()
        if params is not None: self.params.update(params)
        
        self.OmegaK=self.params['OmegaK']
        self.H0=self.params['H0']

        self.Rc=omk2R(self.OmegaK,self.H0)
        self.K=1/self.Rc**2
        
        self.lmax=self.params['lmax']
        self.accboost=self.params['accboost']
        self.verbose = self.params['verbose']

        self.initialize_camb(self.accboost,self.lmax)

        if self.params['compute_kmax_internally']:
            self.get_kmax_from_ell_max()
        else:
            self.kmax = self.params['kmax'] 

        self.nmax = self.k2n(self.kmax)
        self.kk = np.arange(3, self.nmax + 2) / self.Rc
        self.kk[0] += 1e-10

    def n2k(self, n):
        return (n + 1) / self.Rc

    def k2n(self, k):
        return int(np.round(self.Rc * k)) - 1

    def initialize_camb(self, acc_boost, lmax):
        import camb 
        
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.H0,
            ombh2=0.022,
            omch2=0.122,
            mnu=0.06,
            omk=self.OmegaK,
            tau=0.06,
        )
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        pars.set_for_lmax(lmax)

        pars.set_accuracy(
            AccuracyBoost=acc_boost, lAccuracyBoost=acc_boost, lSampleBoost=50
        )
        pars.Accuracy.IntkAccuracyBoost = acc_boost
        pars.Accuracy.SourcekAccuracyBoost = acc_boost
        pars.Accuracy.TransferkBoost = acc_boost
        pars.Accuracy.BesselBoost = acc_boost
        pars.Transfer.high_precision = True

        self.cambpars = pars

    def primpower(self):
        """
        The dimensionless PS for S3: P(k)=k^2/(k^2-K)* As (q/0.05)**(ns-1), where q²=k²-K
        """
        import camb 
        qs = np.sqrt(self.kk**2 - self.K)
        return self.cambpars.scalar_power(qs) * (self.kk**2) / qs**2

    def transfer_functions(self):
        import camb 
        from scipy.interpolate import CubicSpline 

        data = camb.get_transfer_functions(self.cambpars)
        transfer_function = data.get_cmb_transfer_data(tp="scalar")
        transfer_data = np.array(transfer_function.delta_p_l_k) * 1e6 * 2.7255
        k_list = np.array(transfer_function.q)
        ell_list = np.array(transfer_function.L)[: self.lmax - 1] #Has size lmax-1

        interps = CubicSpline(
            k_list,
            transfer_data[0, :ell_list.shape[0], :],
            axis=1,
        )

        interp_transf = interps(self.kk) # Axis 0 has the ell's, Axis 1 has the k's
        return interp_transf
    
    def get_kmax_from_ell_max(self):
        import camb 
        from scipy.interpolate import CubicSpline 
        from threadpoolctl import threadpool_limits
        
        if self.verbose: print('Looking for optimum kmax for the given ell_max...')
        
        # Figure out safe core count for the main thread
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        num_cores = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 4)

        with threadpool_limits(limits=num_cores):
            kmax1 = 1e-4 
            kmax2 = 7e-2
            
            self.params['compute_kmax_internally']=False
            tol = self.params['compute_kmax_tol'] 
            tol2 = 0.01 

            data = camb.get_transfer_functions(self.cambpars)
            transfer_function = data.get_cmb_transfer_data(tp="scalar")
            transfer_data = np.array(transfer_function.delta_p_l_k) * 1e6 * 2.7255
            k_list = np.array(transfer_function.q)
            ell_list = np.array(transfer_function.L)[: self.lmax - 1]

            interps = CubicSpline(
                k_list,
                transfer_data[0, :ell_list.shape[0], :],
                axis=1,
            )
            
            results = camb.get_results(self.cambpars)
            Cls = results.get_cmb_power_spectra(self.cambpars, raw_cl=True, lmax=self.lmax, CMB_unit='muK')['unlensed_scalar']
            Cls_s3 = Cls[2:,0]
             
            while abs(np.log10(kmax2/kmax1)) > tol2:
                
                kmid = (kmax1+kmax2)/2.0
                self.params['kmax']=kmid
                
                self.nmax = self.k2n(kmid)
                self.kk = np.arange(3, self.nmax + 2) / self.Rc
                self.kk[0] += 1e-10

                power = self.primpower()
                interp_transf = interps(self.kk)
                transfer_squared = interp_transf**2
                Cls_s3_guess = 4*pi*transfer_squared @ (power/np.arange(3,self.nmax+2))

                f_k = abs((Cls_s3-Cls_s3_guess)/Cls_s3)
                diff = np.max(f_k)

                if diff > tol:
                    kmax1 = kmid
                else:
                    kmax2 = kmid
                
            if self.verbose: print(f'Convergence reached. kmax is {kmid:1.4e}. '
                  +f'This kmax will compute the Cls with an accuracy of {(1-diff)*100:2.2f}%.')
            self.kmax = kmid

class S3(SphericalTopology):

    def __init__(self,params : dict = None) -> None:
        super().__init__(params)

    def get_Cls_TT(self):
        import camb 
        results = camb.get_results(self.cambpars)
        Cls = results.get_cmb_power_spectra(self.cambpars, raw_cl=True, lmax=self.lmax, CMB_unit='muK')['unlensed_scalar']
        return Cls[2:,0] 
    
    def get_manual_Cls(self):
        power = self.primpower()
        transfer_squared = self.transfer_functions()**2
        return 4*pi*transfer_squared @ (power/np.arange(3,self.nmax+2))
    
class EllipticSpace(SphericalTopology):

    def __init__(self,params : dict = None) -> None:
        super().__init__(params)

    def get_Cls(self):
        power = self.primpower()
        transfer_squared = self.transfer_functions()**2
        return 8*pi*transfer_squared[:,0::2] @ (power[0::2]/np.arange(3,self.nmax+2,2))


class LensSpace(SphericalTopology):

    def __init__(self,params : dict = None) -> None:
        super().__init__(params)

        self.p = self.params['p']
        self.q = self.params['q']

        self.theta0 = self.params['obs_ang'][0]
        self.chi0 = self.params['obs_ang'][1]
        self.phi0 = self.params['obs_ang'][2]

        self.volume=2*pi**2*self.Rc**3/self.p

        if not (isinstance(self.p, int) and isinstance(self.q, int)):
            raise TypeError("Both p and q must be integers.")
        if self.p <= 0 or self.q <= 0:
            raise ValueError("Both p and q must be positive integers.")
        if self.q >= self.p:
            if self.p==1 and self.q==1:
                pass
            else:
                raise ValueError("q must be less than p.")
        if gcd(self.p, self.q) != 1:
            raise ValueError("The greatest common factor of p and q must be 1.")
        
        self.num_workers = self.params['num_workers']
        self.batchsize = self.params['batchsize']
        self.use_tqdm = self.params['use_tqdm']
                
    
    def _compute_n_batch(self, n_list, num_lm, ell_arr, mm_arr,
                          transfer_funcs, power_spectrum, threads_per_worker):
        import traceback
        try:
               
            with threadpool_limits(limits=threads_per_worker):

                C_local= np.zeros((num_lm, num_lm), dtype=np.complex128)

                CHUNK_SIZE = 50000

                for n in n_list:
                    
                    n_idx = n - 2
                    prefactor = power_spectrum[n_idx] / (n + 1)

                    pairs = find_mLmR_pairs(n, self.p, self.q)
                    if len(pairs) == 0: continue

                    mmL_all = (2 * pairs[:, 0]).astype(np.int32)
                    mmR_all = (2 * pairs[:, 1]).astype(np.int32)
                    num_pairs = len(mmL_all)

                    valid_ell_mask = (ell_arr >= 2) & (ell_arr <= n) & ((ell_arr - 2) < transfer_funcs.shape[0]) #transfer_funcs.shape[0] is lmax-1, this is saying ell<lmax+1
                    valid_i_global = np.where(valid_ell_mask)[0]
                    
                    mm_valid = mm_arr[valid_i_global]
                    ell_valid = ell_arr[valid_i_global]
                    delta_valid = np.ascontiguousarray(transfer_funcs[ell_valid - 2, n_idx])

                    if self.chi0 == 0:
                        
                        from scipy.sparse import coo_matrix

                        for k_start in range(0, num_pairs, CHUNK_SIZE):

                            k_end = min(k_start + CHUNK_SIZE, num_pairs)
                            mmL_list = mmL_all[k_start:k_end]
                            mmR_list = mmR_all[k_start:k_end]
                            chunk_pairs_len = len(mmL_list)

                            target_mm = mmL_list + mmR_list
                            valid_mask_2d = (mm_valid[:,None]==target_mm[None,:])
                            flat_i, flat_k = np.where(valid_mask_2d)

                            if len(flat_i) == 0: continue

                            final_i = flat_i.astype(np.int32)
                            final_k = flat_k.astype(np.int32)

                            ell_target = ell_valid[final_i]
                            mm_target = mm_valid[final_i]
                            mmL_target = mmL_list[final_k]
                            
                            #Symmetries of wig3j symbols
                            
                            mm_pos = np.abs(mm_target)
                            flip_mask = (mm_target < 0)
                            mmL_pos = mmL_target.copy()
                            mmL_pos[flip_mask] = -mmL_pos[flip_mask]

                            mmR_pos = mm_pos - mmL_pos
                            swap_mask = (mmR_pos > mmL_pos)
                            mmL_can = np.maximum(mmL_pos, mmR_pos)

                            odd_J = ((n + ell_target) & 1) == 1
                            apply_phase = odd_J & (flip_mask != swap_mask)

                            total_phase = np.ones(len(ell_target), dtype=np.float64)
                            total_phase[apply_phase] = -1.0

                            # The point of doing this weird thing is to do np.unique over a 1D array, which is much faster than
                            # np.unique over a 2D aray and setting axis=0

                            packed_triplets = (
                                ell_target.astype(np.int64) * 1000000000 + 
                                mm_pos.astype(np.int64) * 100000 + 
                                (mmL_can.astype(np.int64) + 20000)
                            )

                            _, unique_indices, inverse_indices = np.unique(
                                packed_triplets, return_index=True, return_inverse=True
                            )


                            w3j_unique = compute_wig3j_flat(
                                n, 
                                ell_target[unique_indices], 
                                mm_pos[unique_indices], 
                                mmL_can[unique_indices]
                            )

                            w3j_vals = w3j_unique[inverse_indices] * total_phase
                            
                            values = delta_valid[final_i] * w3j_vals
                            valid_indices = (valid_i_global[final_i], final_k)

                            # generate sparse matrix
                            V_coo = coo_matrix((values.astype(np.complex128), 
                                                valid_indices), 
                                                shape=(num_lm,chunk_pairs_len), 
                                                dtype=np.complex128)

                            V_csr = V_coo.tocsr()
                            result_sparse = V_csr @ V_csr.conj().T
                            C_local += prefactor * result_sparse.toarray()

                    else:
                        small_d = get_wigner_d_matrix_optimized(n, 2 * self.chi0)

                        for k_start in range(0, num_pairs, CHUNK_SIZE):
                        
                            k_end = min(k_start + CHUNK_SIZE, num_pairs)
                            mmL_list = mmL_all[k_start:k_end]
                            mmR_list = mmR_all[k_start:k_end]
                            chunk_pairs_len = len(mmL_list)

                            diff = mm_valid[:, None] - mmL_list[None, :]
                            valid_mask_2d = (diff >= -n) & (diff <= n)
                            flat_i, flat_k = np.where(valid_mask_2d)
                            
                            row_indices = (mmR_list[flat_k] + n) // 2
                            col_indices = (diff[flat_i, flat_k] + n) // 2
                            flat_d = small_d[row_indices, col_indices]

                            if len(flat_i) == 0: continue

                            final_i = flat_i.astype(np.int32)
                            final_k = flat_k.astype(np.int32)

                            ell_target = ell_valid[final_i]
                            mm_target = mm_valid[final_i]
                            mmL_target = mmL_list[final_k]
                            
                            #Symmetries of wig3j symbols
                            
                            mm_pos = np.abs(mm_target)
                            flip_mask = (mm_target < 0)
                            mmL_pos = mmL_target.copy()
                            mmL_pos[flip_mask] = -mmL_pos[flip_mask]

                            mmR_pos = mm_pos - mmL_pos
                            swap_mask = (mmR_pos > mmL_pos)
                            mmL_can = np.maximum(mmL_pos, mmR_pos)

                            odd_J = ((n + ell_target) & 1) == 1
                            apply_phase = odd_J & (flip_mask != swap_mask)

                            total_phase = np.ones(len(ell_target), dtype=np.float64)
                            total_phase[apply_phase] = -1.0

                            # The point of doing this weird thing is to do np.unique over a 1D array, which is much faster than
                            # np.unique over a 2D aray and setting axis=0

                            packed_triplets = (
                                ell_target.astype(np.int64) * 1000000000 + 
                                mm_pos.astype(np.int64) * 100000 + 
                                (mmL_can.astype(np.int64) + 20000)
                            )

                            _, unique_indices, inverse_indices = np.unique(
                                packed_triplets, return_index=True, return_inverse=True
                            )


                            w3j_unique = compute_wig3j_flat(
                                n, 
                                ell_target[unique_indices], 
                                mm_pos[unique_indices], 
                                mmL_can[unique_indices]
                            )

                            w3j_vals = w3j_unique[inverse_indices] * total_phase

                            V_chunk = np.zeros((num_lm, chunk_pairs_len), dtype=np.complex128)
                            V_chunk[valid_i_global[final_i], final_k] = delta_valid[final_i] * w3j_vals * flat_d

                            C_local += (prefactor * (V_chunk @ V_chunk.conj().T))
                          
            return len(n_list), C_local
        
        except Exception as e:
            print(f"\n--- CRITICAL ERROR IN WORKER ---")
            traceback.print_exc()
            raise e
        
    def compute_Clmlpmp_optimized(self):
        
        start = time.time()
        if self.verbose: print(f'Clmlpmp computation started. nmax is {self.nmax}')
        num_lm = self.lmax * (self.lmax + 2) - 3

        if self.num_workers is not None:
            num_workers = self.num_workers
        else:
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            
            if slurm_cpus is not None:
                num_workers = int(slurm_cpus)
            else:
                try:
                    num_workers = len(os.sched_getaffinity(0))
                except AttributeError:
                    num_workers = os.cpu_count() or 4

            num_workers = int(num_workers//2) 

        total_pool = get_available_cores()

        threads_per_worker = max(1, total_pool // num_workers)

        with threadpool_limits(limits=total_pool):
            power_spectrum = self.primpower()
            transfer_funcs = self.transfer_functions()

        lm_map = np.array([lmindex(i) for i in range(num_lm)])

        ell_arr = np.array([lm[0] for lm in lm_map], dtype=np.int32)
        m_arr = np.array([lm[1] for lm in lm_map], dtype=np.int32)
        mm_arr = 2 * m_arr

        if self.verbose: print(f"Distributing sum in n across {num_workers} parallel CPU cores...")
        
        init_wigner_tables(self.nmax * 2, 3)
        worker_init = init_wigner_tables
        worker_initargs = (self.nmax * 2, 3)

        all_ns_desc = list(range(self.nmax, 1, -1))

        if self.batchsize is not None:
            num_batch = int(np.round(self.nmax/self.batchsize))
        else:
            num_batch = num_workers
        
        n_batches = [[] for _ in range(num_batch)] 
        
        for i, n in enumerate(all_ns_desc):
            n_batches[i % num_batch].append(n)

        n_batches = [batch for batch in n_batches if len(batch) > 0]  

        C_total = np.zeros((num_lm,num_lm),dtype=np.complex128)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=worker_init,
            initargs=worker_initargs
        ) as executor:
            
            futures = [
                executor.submit(
                    self._compute_n_batch, 
                    batch, num_lm, ell_arr, mm_arr,
                    transfer_funcs, power_spectrum, threads_per_worker
                )
                for batch in n_batches
            ]

            completed_n = 0
            total_n = self.nmax-1
            
            if not self.verbose: self.use_tqdm=False

            if not self.use_tqdm:
                for future in concurrent.futures.as_completed(futures):
                    n_count, local_C = future.result() 
                    C_total += local_C
                    completed_n += n_count
                    percent = 100 * completed_n / total_n
                    if self.verbose: print(f"Progress: {completed_n}/{total_n} ({percent:.1f}%)", flush=True)
            else:
                progress_bar = tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures), 
                    desc="Computing batches",
                    smoothing=0.1
                )
                for future in progress_bar:
                    n_count, local_C = future.result() 
                    C_total += local_C
                    completed_n += n_count
                    progress_bar.set_postfix(n_processed=f"{completed_n}/{total_n}")
        
        v=np.zeros(num_lm,dtype=np.complex128)
        for i, (ell,m) in enumerate(lm_map):
            phase = 1 if m%2==0 else -1
            v[i]=1j**(-ell)*sqrt(2*ell+1)*phase*np.exp(-1j*m*(self.theta0-self.phi0))
            
        phase_matrix=np.outer(v,np.conj(v))

        free_wigner_tables()

        total_time = time.time() - start
        if self.verbose: print(f"Time taken: {total_time:.2f}s")

        C_final = self.p * 4 * pi * phase_matrix * C_total

        self.C_matrix = C_final

        transfer_squared = transfer_funcs**2
        Cells = 4*pi*transfer_squared @ (power_spectrum/np.arange(3,self.nmax+2))
        tiled_Cells=np.repeat(Cells,2*np.arange(2,self.lmax+1)+1)
        sqrtClClp = np.sqrt(np.outer(tiled_Cells,tiled_Cells))

        self.norm_C_matrix = C_final/sqrtClClp

    def get_C_matrix(self,norm=True,lmin=2,lmax=10):

        if self.C_matrix is None:
            print('Compute Clmlpmp first')
            return -1
        
        idx_start = nindex(l=lmin,m=-lmin)
        idx_end = nindex(l=lmax,m=lmax)
        if norm:
            return self.norm_C_matrix[idx_start:idx_end+1,idx_start:idx_end+1]
        else:
            return self.C_matrix[idx_start:idx_end+1,idx_start:idx_end+1]

    def compute_KL(self,s3_omk=None,lmin=None,lmax=None):

        if self.norm_C_matrix is None:
            print('Compute Clmlpmp first')
            return -1
        
        if lmin is None:
            lmin = 2
        if lmax is None:
            lmax = self.lmax
        
        if s3_omk is None:
            KL_matrix = self.get_C_matrix(norm=True,lmin=lmin,lmax=lmax)
        else:
            s3_params = self.params.copy()
            s3_params['OmegaK'] = s3_omk
            s3_params['verbose'] = False
            s3_space = S3(s3_params)
            s3_cls = s3_space.get_manual_Cls()
            tiled_s3_cls = np.repeat(s3_cls, 2 * np.arange(2, self.lmax + 1) + 1)
            idx_start = nindex(l=lmin, m=-lmin)
            idx_end = nindex(l=lmax, m=lmax)
            sliced_C_matrix = self.get_C_matrix(norm=False, lmin=lmin, lmax=lmax)
            sliced_tiled_s3 = tiled_s3_cls[idx_start:idx_end+1]
            KL_matrix = sliced_C_matrix / sliced_tiled_s3
        
        lams = np.linalg.eigvals(KL_matrix)
        forward_KL = 0
        backward_KL = 0
        for lam in lams:
            forward_KL += (lam-np.log(lam)-1)
            backward_KL += (1.0/lam+np.log(lam)-1)
        return np.array([np.real(forward_KL),np.real(backward_KL)])


    def plot_Clmlpmp(self, filename=None):
        from matplotlib import pyplot as plt 

        plt.rcParams.update({
            # 1. LaTeX and Fonts
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'text.latex.preamble': r'\usepackage{amsmath, amssymb}', 
            
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            
            
            'axes.linewidth': 1.2,          # Slightly thicker bounding box
            'xtick.direction': 'in',        # Ticks point INWARD
            'ytick.direction': 'in',
            'xtick.top': True,              # Ticks on the top edge
            'ytick.right': True,            # Ticks on the right edge
            'xtick.minor.visible': False,    # Minor ticks are standard in astrophysics
            'ytick.minor.visible': False,
            'xtick.major.size': 6,          # Major tick length
            'xtick.minor.size': 3,          # Minor tick length
            'ytick.major.size': 6,
            'ytick.minor.size': 3,
            'xtick.major.width': 1.0,       # Tick thicknesses
            'ytick.major.width': 1.0,
            
            # 4. Lines and Legend
            'lines.linewidth': 1.5,         # Thick enough to see, thin enough to be precise
            'legend.frameon': True,        
            'legend.loc': 'best',
            
            # 5. Figure Output
            'figure.figsize': (5.0, 5.0),   # Standard aspect ratio for a single column
            'figure.dpi': 150,              
            'savefig.bbox': 'tight',        # Prevents labels from getting cut off
            'savefig.pad_inches': 0.1
        })
        
        plt.figure(figsize=(6,6))
    
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(color='black')
        boundaries = np.cumsum(2*np.arange(2,self.lmax+1)+1) - 0.5

        if self.norm_C_matrix is None:
            print('Compute Clmlpmp first')
            return -1
        
        plt.imshow(np.log10(np.abs(self.norm_C_matrix)),cmap=cmap,vmin=-8,origin='lower')

        ells = np.arange(2, self.lmax + 1)
        counts = 2 * ells + 1
        block_ends = np.cumsum(counts)
        start_indices = np.concatenate(([0], block_ends[:-1]))
        tick_positions = start_indices + counts / 2.0 - 0.5

        boundaries = block_ends - 0.5
        internal_boundaries = boundaries[:-1]
        N = boundaries[-1] + 0.5 
        
        plt.vlines(internal_boundaries, ymin=-0.5, ymax=N-0.5, colors='white', linewidth=0.5, alpha=0.5)
        plt.hlines(internal_boundaries, xmin=-0.5, xmax=N-0.5, colors='white', linewidth=0.5, alpha=0.5)

        ax = plt.gca()
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f'${ell}$' for ell in ells], rotation=0)
        ax.set_xlabel(r'$\ell^\prime$', fontsize=14)
        
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([f'${ell}$' for ell in ells])
        ax.set_ylabel(r'$\ell$', fontsize=14)

        plt.colorbar()
        plt.title(f' L({self.p},{self.q}) \n'+r'$\Omega_K$='+f'{self.OmegaK:.4f} // '+r'$(\theta_0,\chi_0,\varphi_0)=$'+f'({self.theta0:.2f},{self.chi0:.2f},{self.phi0:.2f})',fontsize=12)
        if filename is not None:
            plt.savefig(filename)
