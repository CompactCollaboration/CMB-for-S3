import itertools
import os
import time
import concurrent.futures
import numpy as np
from math import pi, gcd, sqrt
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from s3tools import *
from wigner_d_jacobi import get_wigner_d_matrix_optimized
from fast_wigner import init_wigner_tables, free_wigner_tables, compute_wig3j_flat

class SphericalTopology_params():
    def __init__(self,params : dict = None) -> None:
        self.params = get_default_parameters()
        if params is not None: self.params.update(params)

        self.OmegaK = np.atleast_1d(self.params['OmegaK'])
        self.H0 = self.params['H0']
        self.Rc = omk2R(self.OmegaK,self.H0)
        self.K = 1/self.Rc**2

        self.lmax = self.params['lmax']

        self.accboost= self.params['accboost']
        self.verbose = self.params['verbose']
        self.use_tqdm = self.params['use_tqdm']
        self.num_workers = self.params['num_workers']
        self.batchsize = self.params['batchsize']

        # here, we want the params that give us the largest nmax
        self.OmegaK_opt = np.min(self.OmegaK)
        self.Rc_opt = np.max(self.Rc)
        self.K_opt = np.min(self.K)

        self.cambpars_opt = self.initialize_camb(self.accboost, self.lmax, self.OmegaK_opt)

        # self.kmax and self.nmax will be the same for all values
        if self.params['compute_kmax_internally']:
            self.get_kmax_from_ell_max_opt()
        else:
            self.kmax = self.params['kmax'] 

        self.nmax = self.k2n(self.kmax)
        self.kk = np.arange(3, self.nmax + 2) / self.Rc[:,np.newaxis]
        self.kk[0,:] += 1e-10 # self.kk has shape (nmax+1, OmegaK) 

        self.kk_opt = np.arange(3, self.nmax + 2) / self.Rc_opt
        self.kk_opt[0] += 1e-10

    def n2k(self, n):
        return (n + 1) / self.Rc_opt

    def k2n(self, k):
        return int(np.round(self.Rc_opt * k)) - 1

    def initialize_camb(self, acc_boost, lmax, OmegaK):
        import camb 
        
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.H0,
            ombh2=0.022,
            omch2=0.122,
            mnu=0.06,
            omk=OmegaK,
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

        return pars
    
    def primpower(self, OmegaK):
        """
        The dimensionless PS for S3: P(k)=k^2/(k^2-K)* As (q/0.05)**(ns-1), where q²=k²-K
        """
        Rc = omk2R(OmegaK,self.H0)
        K = 1/Rc**2
        kk = np.arange(3, self.nmax + 2) / Rc
        kk[0] += 1e-10

        qs = np.sqrt(kk**2 - K)
        return self.cambpars_opt.scalar_power(qs) * (kk**2) / qs**2

    def transfer_functions(self, OmegaK, onlyTT=True):
        import camb 
        from scipy.interpolate import CubicSpline 

        cambpars = self.initialize_camb(self.accboost, self.lmax, OmegaK)
        Rc = omk2R(OmegaK,self.H0)
        kk = np.arange(3, self.nmax + 2) / Rc
        kk[0] += 1e-10

        data = camb.get_transfer_functions(cambpars)
        transfer_function = data.get_cmb_transfer_data(tp="scalar")
        transfer_data = np.array(transfer_function.delta_p_l_k) * 1e6 * 2.7255
        k_list = np.array(transfer_function.q)
        ell_list = np.array(transfer_function.L)[: self.lmax - 1] #Has size lmax-1

        interps_T = CubicSpline(
            k_list,
            transfer_data[0, :ell_list.shape[0], :],
            axis=1,
        )
        interp_transf_T = interps_T(kk) # Axis 0 has the ell's, Axis 1 has the k's

        if onlyTT:
            return interp_transf_T
        else:
            interps_E = CubicSpline(
                k_list,
                transfer_data[1, :ell_list.shape[0], :],
                axis=1,
            )
            interp_transf_E = interps_E(kk)
            spin2_prefactor = np.sqrt(ell_list*(ell_list+1)*(ell_list-1)*(ell_list+2))
            return interp_transf_T, interp_transf_E*spin2_prefactor[:,np.newaxis]   

    def get_kmax_from_ell_max_opt(self):
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

            data = camb.get_transfer_functions(self.cambpars_opt)
            transfer_function = data.get_cmb_transfer_data(tp="scalar")
            transfer_data = np.array(transfer_function.delta_p_l_k) * 1e6 * 2.7255
            k_list = np.array(transfer_function.q)
            ell_list = np.array(transfer_function.L)[: self.lmax - 1]

            interps = CubicSpline(
                k_list,
                transfer_data[0, :ell_list.shape[0], :],
                axis=1,
            )
            
            results = camb.get_results(self.cambpars_opt)
            Cls = results.get_cmb_power_spectra(self.cambpars_opt, raw_cl=True, 
                                                lmax=self.lmax, CMB_unit='muK')['unlensed_scalar']
            Cls_s3 = Cls[2:,0]
             
            while abs(np.log10(kmax2/kmax1)) > tol2:
                
                kmid = (kmax1+kmax2)/2.0
                self.params['kmax']=kmid
                
                self.nmax = self.k2n(kmid)
                self.kk_opt = np.arange(3, self.nmax + 2) / self.Rc_opt
                self.kk_opt[0] += 1e-10

                power = self.primpower(self.OmegaK_opt)
                interp_transf = interps(self.kk_opt)
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

    def get_cosmologies(self,onlyTT=True):
        """
        Computes power spectra and transfer functions for all given omegaK and lmax values.
        The reason this is done in one function is so that we only have to initialize camb once per 
        (lmax, OmegaK) pair.

        Returns an array of size (omegak_arr*lmax_arr, 4), where the first two columns give the 
        omegaK and lmax values, and the last two columns give the corresponding power spectra and 
        transfer function arrays.
        """
        import camb 
        from scipy.interpolate import CubicSpline 

        all_cosmologies = np.empty((0, 3), dtype=object)

        for omk in self.OmegaK:
            print(f"--- Computing Cosmology: OmegaK={omk} ---",flush=True)

            # Initialize a base topology just to get the transfer functions
            total_pool = get_available_cores()

            with threadpool_limits(limits=total_pool):

                power_spectrum = self.primpower(omk)

                if onlyTT:
                    transfer_funcs = self.transfer_functions(omk, onlyTT=onlyTT)
                else:
                    transf_T, transf_E = self.transfer_functions(omk, onlyTT=onlyTT)
                    transfer_funcs = np.stack([transf_T, transf_E], axis=0)
            
            new_row = np.array([[omk, power_spectrum, transfer_funcs]], dtype=object)
            all_cosmologies = np.vstack([all_cosmologies, new_row])
        
        return all_cosmologies
    

class LensSpace_params(SphericalTopology_params):
    def __init__(self,params : dict = None) -> None:
        super().__init__(params)

        self.p = np.atleast_1d(self.params['p'])
        self.q = np.atleast_1d(self.params['q'])

        # Handle obs_ang specifically to ensure it's a list of tuples/lists
        self.obs_ang = [self.params['obs_ang']] if np.ndim(self.params['obs_ang']) == 1 else self.params['obs_ang']

    def p_q_allowed(self,p,q):
        if not (isinstance(p, (int,np.int32,np.int64)) and isinstance(q, (int,np.int32,np.int64))):
            raise TypeError(f"Both p and q must be integers. p={p}, q={q}.")
        if p <= 0 or q <= 0:
            raise ValueError(f"Both p and q must be positive integers. p={p}, q={q}.")
        if q >= p:
            if p==1 and q==1:
                pass
            else:
                raise ValueError(f"q must be less than p. p={p}, q={q}.")
        if gcd(p, q) != 1:
            raise ValueError(f"The greatest common factor of p and q must be 1. p={p}, q={q}.")
    
    def _compute_n_batch_single(self, n_list, topology_pars, num_lm, ell_arr, mm_arr,
                          transfer_funcs, power_spectrum, threads_per_worker,onlyTT=True):
        """
        Compute matrix for a single configuration (one set of p, q, obs_ang, and OmegaK)
        """
        import traceback

        (p,q,obs_ang) = topology_pars

        chi0 = obs_ang[1]
        if chi0 == 0 or q == 1:
            from scipy.sparse import coo_matrix

        try:
            with threadpool_limits(limits=threads_per_worker):
                N_fields = 1 if onlyTT else 2
                C_local= np.zeros((N_fields*num_lm, N_fields*num_lm), dtype=np.complex128)

                CHUNK_SIZE = 50000

                for n in n_list:
                    
                    n_idx = n - 2
                    prefactor = power_spectrum[n_idx] / (n + 1)

                    pairs = find_mLmR_pairs(n, p, q)
                    if len(pairs) == 0: continue

                    mmL_all = (2 * pairs[:, 0]).astype(np.int32)
                    mmR_all = (2 * pairs[:, 1]).astype(np.int32)
                    num_pairs = len(mmL_all)

                    valid_ell_mask = (ell_arr >= 2) & (ell_arr <= n) & ((ell_arr - 2) < (transfer_funcs.shape[1] if not onlyTT else transfer_funcs.shape[0])) #transfer_funcs.shape[0] is lmax-1, this is saying ell<lmax+1
                    valid_i_global = np.where(valid_ell_mask)[0]
                    
                    mm_valid = mm_arr[valid_i_global]
                    ell_valid = ell_arr[valid_i_global]

                    if onlyTT:
                        delta_valid = np.ascontiguousarray(transfer_funcs[ell_valid - 2, n_idx])
                    else:
                        delta_valid_T = np.ascontiguousarray(transfer_funcs[0, ell_valid - 2, n_idx])
                        delta_valid_E = np.ascontiguousarray(transfer_funcs[1, ell_valid - 2, n_idx])

                    if chi0 == 0 or q == 1:
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
                            
                            if onlyTT:
                                values = delta_valid[final_i] * w3j_vals
                                valid_indices = (valid_i_global[final_i], final_k)
                                V_coo = coo_matrix((values.astype(np.complex128), valid_indices), 
                                                    shape=(num_lm, chunk_pairs_len), dtype=np.complex128)
                                V_csr = V_coo.tocsr()
                                C_local += prefactor * (V_csr @ V_csr.conj().T).toarray()
                            else:
                                values_T = delta_valid_T[final_i] * w3j_vals
                                values_E = delta_valid_E[final_i] * w3j_vals
                                
                                row_idx_T = valid_i_global[final_i]
                                row_idx_E = valid_i_global[final_i] + num_lm

                                values_joint = np.concatenate([values_T, values_E])
                                row_joint = np.concatenate([row_idx_T, row_idx_E])
                                col_joint = np.concatenate([final_k, final_k])

                                Z_coo = coo_matrix((values_joint.astype(np.complex128), (row_joint, col_joint)), 
                                                    shape=(2*num_lm, chunk_pairs_len), dtype=np.complex128)
                                Z_csr = Z_coo.tocsr()
                                C_local += prefactor * (Z_csr @ Z_csr.conj().T).toarray()

                    else:
                        small_d = get_wigner_d_matrix_optimized(n, 2 * chi0)

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

                            common_term = w3j_vals * flat_d

                            if onlyTT:
                                V_chunk = np.zeros((num_lm, chunk_pairs_len), dtype=np.complex128)
                                V_chunk[valid_i_global[final_i], final_k] = delta_valid[final_i] * common_term
                                C_local += (prefactor * (V_chunk @ V_chunk.conj().T))
                            else:
                                Z_chunk = np.zeros((2*num_lm, chunk_pairs_len), dtype=np.complex128)
                                Z_chunk[valid_i_global[final_i], final_k] = delta_valid_T[final_i] * common_term
                                Z_chunk[valid_i_global[final_i] + num_lm, final_k] = delta_valid_E[final_i] * common_term
                                C_local += (prefactor * (Z_chunk @ Z_chunk.conj().T))
                          
            return len(n_list), C_local
        
        except Exception as e:
            print(f"\n--- CRITICAL ERROR IN WORKER ---")
            traceback.print_exc()
            raise e
    
    def compute_Clmlpmp_single(self, topology_pars, transfer_funcs, power_spectrum, onlyTT=True):
        """
        Compute the full matrix for only one set topology parameters.
        """
        (p, q, obs_ang) = topology_pars
        (theta0,chi0,phi0) = obs_ang

        start = time.time()
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

        N_fields = 1 if onlyTT else 2
        C_total = np.zeros((N_fields * num_lm, N_fields * num_lm), dtype=np.complex128)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=worker_init,
            initargs=worker_initargs
            ) as executor:

            
            futures = [
                executor.submit(
                    self._compute_n_batch_single, 
                    batch, [int(p),int(q),obs_ang], num_lm, ell_arr, mm_arr,
                    transfer_funcs, power_spectrum, threads_per_worker, onlyTT
                )
                    for batch in n_batches
                ]

            completed_n = 0
                
            for future in concurrent.futures.as_completed(futures):
                n_count, local_C = future.result() 
                C_total += local_C
                completed_n += n_count
            
            v=np.zeros(num_lm,dtype=np.complex128)
            for i, (ell,m) in enumerate(lm_map):
                phase = 1 if m%2==0 else -1
                v[i]=1j**(-ell)*sqrt(2*ell+1)*phase*np.exp(-1j*m*(theta0-phi0))
                
            phase_matrix=np.outer(v,np.conj(v))

            if not onlyTT:
                phase_matrix = np.block([
                    [phase_matrix, phase_matrix],
                    [phase_matrix, phase_matrix]
                ])

            free_wigner_tables()

            total_time = time.time() - start
            if self.verbose:
                print(f"Time taken: {total_time // 60} mins {total_time % 60} secs.")

            C_final = p * 4 * pi * phase_matrix * C_total

            return C_final

    def compute_Clmlpmp_optimized_all(self,onlyTT=True):

        unique_topologies = list(itertools.product(self.p, self.q, self.obs_ang))
        all_cosmologies = self.get_cosmologies(onlyTT=onlyTT)
        total_num = len(unique_topologies)*len(all_cosmologies)
        
        self.all_norm_C_matrix = {}
        self.all_C_matrix = {}

        progress_bar = tqdm(total=total_num, 
                    desc="Computing topologies",
                    smoothing=0.1)

        with progress_bar as pbar:
            # loop over curvature
            for i, (OmegaK, power_spectrum, transfer_funcs) in enumerate(all_cosmologies):

                # calculate normalization first for all lens spaces of same curvature
                if onlyTT:
                    transfer_squared = transfer_funcs**2
                    Cells = 4*pi*transfer_squared @ (power_spectrum/np.arange(3,self.nmax+2))
                    tiled_Cells = np.repeat(Cells, 2*np.arange(2,self.lmax+1)+1)
                    sqrtClClp = np.sqrt(np.outer(tiled_Cells, tiled_Cells))
                else:
                    transfer_squared_T = transfer_funcs[0]**2
                    transfer_squared_E = transfer_funcs[1]**2
                    
                    Cells_TT = 4*pi*transfer_squared_T @ (power_spectrum/np.arange(3,self.nmax+2))
                    Cells_EE = 4*pi*transfer_squared_E @ (power_spectrum/np.arange(3,self.nmax+2))
                    
                    tiled_TT = np.repeat(Cells_TT, 2*np.arange(2,self.lmax+1)+1)
                    tiled_EE = np.repeat(Cells_EE, 2*np.arange(2,self.lmax+1)+1)
                        
                    tiled_joint = np.concatenate([tiled_TT, tiled_EE])
                    sqrtClClp = np.sqrt(np.outer(tiled_joint, tiled_joint))

                # loop over lens spaces
                for j, (p, q, obs_ang) in enumerate(unique_topologies):
                    try:
                        self.p_q_allowed(p,q)
                    except Exception as e:
                        print(f'{e}. Skipping computation.')
                        continue

                    key = (OmegaK, p, q,tuple(obs_ang))
                    chi0 = obs_ang[1]

                    if self.verbose:
                        print(f'Computing matrix for OmegaK={OmegaK}, nmax={self.nmax}, p={p}, q={q}, chi0={chi0}')


                    C_total = self.compute_Clmlpmp_single((int(p),int(q),obs_ang), 
                                                          transfer_funcs,
                                                          power_spectrum,
                                                          onlyTT)

                    self.all_C_matrix[key] = C_total
                    self.all_norm_C_matrix[key] = C_total/sqrtClClp
                    
                    completed_num = (i+1)*(j+1)

                    if self.use_tqdm:
                        pbar.set_postfix(n_processed=f"{completed_num}/{total_num}")
                        

    def get_C_matrix(self, pars, norm=True, lmin=None, lmax=None):
        if self.all_C_matrix is None:
            print('Compute Clmlpmp first')
            return -1
        
        if lmin is None: lmin = 2
        if lmax is None: lmax = self.lmax
        
        num_lm = self.lmax * (self.lmax + 2) - 3
        idx_start = nindex(l=lmin, m=-lmin)
        idx_end = nindex(l=lmax, m=lmax)
        
        key = (pars[0], pars[1], pars[2], tuple(pars[3]))
        matrix = self.all_norm_C_matrix[key] if norm else self.all_C_matrix[key]
        
        is_joint = (matrix.shape[0] == 2 * num_lm)

        if is_joint:
            indices = np.concatenate([
                np.arange(idx_start, idx_end + 1),
                np.arange(idx_start, idx_end + 1) + num_lm
            ])
            return matrix[np.ix_(indices, indices)]
        else:
            return matrix[idx_start:idx_end+1, idx_start:idx_end+1]

    def compute_KL_all(self, s3_omk=None, lmin=None, lmax=None):

        if self.all_norm_C_matrix is None:
            print('Compute Clmlpmp first')
            return -1
        
        if lmin is None: lmin = 2
        if lmax is None: lmax = self.lmax
        
        num_lm = self.lmax * (self.lmax + 2) - 3

        unique_topologies = list(itertools.product(self.p, self.q, self.obs_ang))

        is_joint = (next(iter(self.all_norm_C_matrix.values())).shape[0] == 2 * num_lm)
        
        all_KL = {}

        if s3_omk is not None:
                power_spectrum = self.primpower(s3_omk)
                if not is_joint:
                    transfer_funcs = self.transfer_functions(s3_omk, onlyTT=True)
                else:
                    transf_T, transf_E = self.transfer_functions(s3_omk, onlyTT=False)
                    transfer_funcs = np.stack([transf_T, transf_E], axis=0)
        else:
            if is_joint:
                all_cosmologies = self.get_cosmologies(onlyTT=False)
            else:
                all_cosmologies = self.get_cosmologies(onlyTT=True)
            
        # loop over curvature
        for i,OmegaK in enumerate(self.OmegaK):
            if s3_omk is None:
                power_spectrum = all_cosmologies[i,1]
                transfer_funcs = all_cosmologies[i,2]

            n_arr = np.arange(3, self.nmax+2)
        
            idx_start = nindex(l=lmin, m=-lmin)
            idx_end = nindex(l=lmax, m=lmax)

            for j, (p, q, obs_ang) in enumerate(unique_topologies):
                pars = (OmegaK, p, q, tuple(obs_ang))
                sliced_C_matrix = self.get_C_matrix(pars, norm=False, lmin=lmin, lmax=lmax)

            
                if not is_joint:
                    s3_TT = 4*pi*(transfer_funcs**2) @ (power_spectrum/n_arr)
                    tiled_s3 = np.repeat(s3_TT, 2 * np.arange(2, self.lmax + 1) + 1)
                
                    slice_TT = tiled_s3[idx_start:idx_end+1]
                    KL_matrix =  sliced_C_matrix @ np.diag(1.0 / slice_TT)
                
                else:
                    s3_TT = 4*pi*(transfer_funcs[0]**2) @ (power_spectrum/n_arr)
                    s3_EE = 4*pi*(transfer_funcs[1]**2) @ (power_spectrum/n_arr)
                    s3_TE = 4*pi*(transfer_funcs[0] * transfer_funcs[1]) @ (power_spectrum/n_arr)
                
                    counts = 2 * np.arange(2, self.lmax + 1) + 1
                    tiled_TT = np.repeat(s3_TT, counts)
                    tiled_EE = np.repeat(s3_EE, counts)
                    tiled_TE = np.repeat(s3_TE, counts)
                    
                    slice_TT = tiled_TT[idx_start:idx_end+1]
                    slice_EE = tiled_EE[idx_start:idx_end+1]
                    slice_TE = tiled_TE[idx_start:idx_end+1]
                    
                    S3_cov_sliced = np.block([
                        [np.diag(slice_TT), np.diag(slice_TE)],
                        [np.diag(slice_TE), np.diag(slice_EE)]
                    ])
                    
                    KL_matrix = sliced_C_matrix @ np.linalg.inv(S3_cov_sliced)

                lams = np.linalg.eigvals(KL_matrix)
                forward_KL = 0
                backward_KL = 0
                
                for lam in lams:
                    forward_KL += (lam - np.log(lam) - 1)
                    backward_KL += (1.0/lam + np.log(lam) - 1)

                all_KL[pars] = np.array([np.real(forward_KL), np.real(backward_KL)])
            
        return all_KL

