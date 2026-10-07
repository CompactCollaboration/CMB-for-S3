import os
from time import time
import concurrent.futures
import numpy as np
from math import pi, gcd, sqrt
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from s3tools import *
from wigner_d_jacobi import get_wigner_d_matrix_optimized
from fast_wigner import init_wigner_tables, free_wigner_tables, compute_wig3j_flat

class SphericalTopology():

    def __init__(self,params : dict = None) -> None:
        self.params = get_default_parameters()

        if params is not None:
            # check for unknown parameter names
            unknown = set(params) - set(self.params)
            if unknown:
                raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
            
            self.params.update(params)


        # consistency checks for input params:
        for name,val in self.params.items():
            if name in ['OmegaK','compute_kmax_tol','ombh2','omch2','mnu','tau','H0','As','ns']:
                if not (isinstance(val,(float,np.floating))):
                    if name == 'H0' and not (isinstance(val,(int,np.integer))):
                        raise TypeError(f"{name} must be a number.")
                    else:   
                        raise TypeError(f"{name} must be a number.")
                if not np.isfinite(val):
                    raise ValueError(f"{name} must be a finite number.")

                if name == 'OmegaK' and val >= 0:
                    raise ValueError("OmegaK must be a negative number.")
                
                if name == 'compute_kmax_tol' and (val <= 0 or val >= 1):
                    raise ValueError("compute_kmax_tol must be a value between 0 and 1.")

                if name in ['ombh2','omch2','mnu','tau'] and val < 0:
                    raise ValueError("OmegaK must be a non-negative number.")
                
                if name in ['H0','As']:
                    if val <= 0:
                        raise ValueError(f"{name} must be a positive number.")
                    
            if name in ['lmax','accboost']:
                if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)):
                    raise TypeError(f"{name} must be an integer.")
                if name == 'lmax' and val <2:
                    raise ValueError("lmax must be at least 2.")

                if name == 'accboost' and val<1:
                    raise ValueError("accboost must be at least 1.")

                val = int(val)
                self.params[name] = val

            if name in ['num_workers','batchsize']:
                if val is None:
                    continue
                if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)):
                    raise TypeError(f"{name} must be None or an integer.")

                if val < 1:
                    raise ValueError(f"{name} must be at least 1.")

                if name == 'num_workers' and val > get_available_cores():
                    raise ValueError("num_workers exceeds the available CPU allocation.")

                val = int(val)
                self.params[name] = val

            if name in ['p','q']:
                if not isinstance(val, (int, np.integer)):
                    raise TypeError(f"{name} must be an integer.")

                if val <=0:
                    raise ValueError(f"{name} must be a positive integer.")
                
                val = int(val)
                self.params[name] = val
                
                
            if name in ['compute_kmax_internally','use_tqdm','verbose','onlyTT']:
                if not (isinstance(val,(bool,np.bool_))):
                    raise TypeError(f"{name} must be True or False.")
                if name == 'compute_kmax_internally' and not val:
                    if not (isinstance(self.params['kmax'],float)):
                        raise TypeError("kmax must be a number.")
                    if (not np.isfinite(self.params['kmax'])) or self.params['kmax'] <=0:
                        raise ValueError("kmax must be a finite positive number.")

                
        # lens space angle consistency checks:
        angles = np.asarray(self.params['obs_ang'])
        if (angles.shape != (3,)
            or angles.dtype.kind not in 'iuf'  # integer, unsigned, or float
            or not np.all(np.isfinite(angles))):
            raise ValueError("obs_ang must contain exactly three finite real numbers.")

        self.chi0, self.xi_p0, self.xi_m0 = angles

        if not 0 <= self.chi0 <= np.pi / 2:
            raise ValueError("chi0 must be between 0 and pi/2 radians.")

        if not (0 <= self.xi_p0 <= 2*np.pi and 0 <= self.xi_m0 <= 2*np.pi):
            raise ValueError("xi_p0 and xi_m0 must be between 0 and 2*pi radians.")
    
        # lens space p and q consistency checks:
        self.p = self.params['p']
        self.q = self.params['q']

        if self.q >= self.p:
            if self.p==1 and self.q==1:
                pass
            else:
                raise ValueError("q must be less than p.")
        if gcd(self.p, self.q) != 1:
            raise ValueError("The greatest common factor of p and q must be 1.")
        

        # set cosmological parameters
        self.OmegaK=self.params['OmegaK']
        self.H0=self.params['H0']
        self.ombh2=self.params['ombh2']
        self.omch2=self.params['omch2']
        self.mnu=self.params['mnu']
        self.tau=self.params['tau']
        self.As=self.params['As']
        self.ns=self.params['ns']

        # set curvature parameters
        self.Rc=omk2R(self.OmegaK,self.H0)
        self.K=1/self.Rc**2
        self.lmax=self.params['lmax']

        # optional settings
        self.verbose = self.params['verbose']
        self.num_workers = self.params['num_workers']
        self.batchsize = self.params['batchsize']
        self.use_tqdm = self.params['use_tqdm']
        self.onlyTT = self.params['onlyTT']

        # camb settings
        self.accboost=self.params['accboost']
        self.initialize_camb()

        # set kmax
        if self.params['compute_kmax_internally']:
            self.kmax = self.get_kmax_from_lmax()
        else:
            self.kmax = self.params['kmax']
            self.nmax = k2n(self.kmax,self.Rc)
            
            if self.nmax < 2:
                raise ValueError(f"kmax={self.kmax:g} gives no modes with n >= 2. "
                    f"Increase kmax to at least {3 / self.Rc:g} Mpc^-1.")

            self.kk = np.arange(3, self.nmax + 2) / self.Rc
            self.kk[0] += 1e-10

        self.C_matrix = None
        self.norm_C_matrix = None

    def initialize_camb(self):
        """
        Set cosmological parameters using CAMB
        """
        import camb 
        
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.H0,
            ombh2=self.ombh2,
            omch2=self.omch2,
            mnu=self.mnu,
            omk=self.OmegaK,
            tau=self.tau,
        )
        pars.InitPower.set_params(As=self.As, ns=self.ns, r=0)
        pars.set_for_lmax(self.lmax)

        pars.set_accuracy(
            AccuracyBoost=self.accboost, lAccuracyBoost=self.accboost, lSampleBoost=50
        )
        pars.Accuracy.IntkAccuracyBoost = self.accboost
        pars.Accuracy.SourcekAccuracyBoost = self.accboost
        pars.Accuracy.TransferkBoost = self.accboost
        pars.Accuracy.BesselBoost = self.accboost
        pars.Transfer.high_precision = True

        self.cambpars = pars

    def primpower(self):
        """
        The dimensionless primordial power spectrum for S3:
        P(k)=k^2/(k^2-K)* As (q/0.05)**(ns-1), where q^2=k^2-K.
        """
        qs = np.sqrt(self.kk**2 - self.K)
        return self.cambpars.scalar_power(qs) * (self.kk**2) / qs**2

    def transfer_functions(self):
        """
        Calculate transfer functions.
        """
        import camb 
        from scipy.interpolate import CubicSpline 

        data = camb.get_transfer_functions(self.cambpars)
        transfer_function = data.get_cmb_transfer_data(tp="scalar")
        transfer_data = np.array(transfer_function.delta_p_l_k) * 1e6 * 2.7255
        k_list = np.array(transfer_function.q)
        ell_list = np.array(transfer_function.L)[: self.lmax - 1] #Has size lmax-1

        interps_T = CubicSpline(
            k_list,
            transfer_data[0, :ell_list.shape[0], :],
            axis=1,
        )
        interp_transf_T = interps_T(self.kk) # Axis 0 has the ell's, Axis 1 has the k's

        if self.onlyTT:
            return interp_transf_T
        else:
            interps_E = CubicSpline(
                k_list,
                transfer_data[1, :ell_list.shape[0], :],
                axis=1,
            )
            interp_transf_E = interps_E(self.kk)
            spin2_prefactor = np.sqrt(ell_list*(ell_list+1)*(ell_list-1)*(ell_list+2))
            return interp_transf_T, interp_transf_E*spin2_prefactor[:,np.newaxis]
    
    def get_kmax_from_lmax(self,kmax1=1e-4,kmax2=7e-2,kmax_tol=0.01, max_iter=100):
        """
        Calculate kmax internally given chosen lmax using bisection method.
        Parameters
        ----------
        kmax1 : float, optional
            Lower bound of kmax.
        kmax2 : float, optional
            Upper bound of kmax.
        kmax_tol: float, optional
            Maximum difference in log(kmax2) and log(kmax1).
            Determines precision in the bisection method.
        max_iter : int, optional
            Max iterations for bisection method.
        """
        import camb 
        from scipy.interpolate import CubicSpline 
        from threadpoolctl import threadpool_limits

        tol = self.params["compute_kmax_tol"]

        # consistency checks for inputs
        if not np.isfinite(tol) or not 0 < tol < 1:
            raise ValueError("compute_kmax_tol must be finite and strictly between 0 and 1.")

        if (isinstance(max_iter, (bool, np.bool_)) 
            or not isinstance(max_iter, (int, np.integer)) 
            or max_iter < 1):

            raise ValueError("max_iter must be a positive integer.")

        if not np.isfinite(kmax_tol) or kmax_tol <= 0:
            raise ValueError("kmax_tol must be finite and positive.")
        kmax1 = max(kmax1, 3 / self.Rc)
        if not (np.isfinite(kmax1) and np.isfinite(kmax2) and 0 < kmax1 < kmax2):
            raise ValueError(
                "The kmax search requires an upper bound above "
                "both the lower bound and the first mode, 3/Rc."
            )
        
        if self.verbose: print("Looking for optimum kmax for the given lmax...")
        
        # Figure out safe core count for the main thread
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        num_cores = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 4)

        with threadpool_limits(limits=num_cores):
            

            data = camb.get_transfer_functions(self.cambpars)
            transfer_function = data.get_cmb_transfer_data(tp="scalar")
            transfer_data = np.array(transfer_function.delta_p_l_k) * 1e6 * 2.7255
            k_list = np.array(transfer_function.q)
            ell_list = np.array(transfer_function.L)[: self.lmax - 1] #Has size lmax-1

            interps_T = CubicSpline(
                k_list,
                transfer_data[0, :ell_list.shape[0], :],
                axis=1,
            )
            
            results = camb.get_results(self.cambpars)
            Cls = results.get_cmb_power_spectra(self.cambpars, raw_cl=True, lmax=self.lmax, 
                                                CMB_unit='muK')['unlensed_scalar']
            
            if self.onlyTT:
                Cls_s3 = Cls[2:,0]
            else:
                Cls_s3_TT = Cls[2:,0]
                Cls_s3_EE = Cls[2:,1]
                Cls_s3_TE = Cls[2:,3]

                interps_E = CubicSpline(
                    k_list,
                    transfer_data[1, :ell_list.shape[0], :],
                    axis=1,
                )
                spin2_prefactor = np.sqrt(ell_list*(ell_list+1)*(ell_list-1)*(ell_list+2))

            # ensure the loop evaluates at least once
            def error_at(cutoff):
                nmax = k2n(cutoff, self.Rc)
                modes = np.arange(3, nmax + 2)
                kk = modes / self.Rc

                if kk.size == 0:
                    raise RuntimeError(
                        f"No usable modes at kmax={cutoff:.6g}."
                    )

                kk[0] += 1e-10

                # Same calculation as primpower(), using a local trial grid.
                qs = np.sqrt(kk**2 - self.K)
                power = self.cambpars.scalar_power(qs) * kk**2 / qs**2
                ps_factor = power / modes

                T = interps_T(kk)
                guess_TT = 4 * pi * ((T * T) @ ps_factor)

                # Nonfinite errors are checked explicitly below.
                with np.errstate(divide="ignore", invalid="ignore"):
                    if self.onlyTT:
                        errors = np.abs(
                            (Cls_s3 - guess_TT) / Cls_s3
                        )
                    else:
                        E = interps_E(kk) * spin2_prefactor[:, None]
                        guess_EE = 4 * pi * ((E * E) @ ps_factor)
                        guess_TE = 4 * pi * ((T * E) @ ps_factor)

                        errors = np.array([
                            np.abs(
                                (Cls_s3_TT - guess_TT) / Cls_s3_TT
                            ),
                            np.abs(
                                (Cls_s3_EE - guess_EE) / Cls_s3_EE
                            ),
                            np.abs(
                                (Cls_s3_TE - guess_TE) / Cls_s3_TE
                            ),
                        ])

                if not np.all(np.isfinite(errors)):
                    raise RuntimeError(
                        f"Nonfinite spectrum error at kmax={cutoff:.6g}; "
                        "check the computed spectra and reference denominators."
                    )

                return float(np.max(errors))

            # If the lowest allowed cutoff passes, use it immediately.
            best_k = kmax1
            best_error = error_at(best_k)

            if best_error > tol:
                # Bisection requires a tested, passing upper endpoint.
                best_k = kmax2
                best_error = error_at(best_k)

                if best_error > tol:
                    raise RuntimeError(
                        f"Upper search bound kmax={kmax2:.6g} failed: "
                        f"maximum relative error={best_error:.6g}, "
                        f"required <= {tol:.6g}."
                    )

                for _ in range(max_iter):
                    if np.log10(kmax2 / kmax1) <= kmax_tol:
                        break

                    kmid = (kmax1 + kmax2) / 2.0
                    diff = error_at(kmid)

                    if diff <= tol:
                        kmax2 = kmid
                        best_k, best_error = kmid, diff
                    else:
                        kmax1 = kmid

                if np.log10(kmax2 / kmax1) > kmax_tol:
                    raise RuntimeError(
                        "Cutoff search did not reach interval precision "
                        f"within {max_iter} iterations."
                    )

            # Commit the tested, passing result and its matching mode grid.
            self.kmax = best_k
            self.params["kmax"] = best_k
            self.params["compute_kmax_internally"] = False
            self.nmax = k2n(best_k, self.Rc)
            self.kk = np.arange(3, self.nmax + 2) / self.Rc
            self.kk[0] += 1e-10

            if self.verbose:
                print(f"Convergence reached. kmax is {best_k:.6g}. nmax is {self.nmax}")
                print(f"This kmax will compute the Cls with an accuracy of {(1-best_error)*100:2.3f}% "
                     f"(target >= {(1-tol)*100:2.3f}%).")
            return best_k

    def get_C_matrix(self, norm=True, lmin=None, lmax=None):
        """
        Returns a slice of the CMB correlation matrix given specified lmin and lmax.
        Parameters
        ----------
        norm : bool
            If True, returns matrix normalized by sqrt(Cl*Clp) from S3.
            If False, returns non-normalized matrix.
        lmin : int or None, optional
            min l to return matrix. If None, set to lmin=2.
        lmax : int or None, optional
            max l to return matrix. If None, set to self.lmax.
        """
        if self.C_matrix is None or self.norm_C_matrix is None:
            raise RuntimeError("The correlation matrix has not been computed. Call compute_Clmlpmp() first.")

        if lmin is None: lmin = 2
        if lmax is None: lmax = self.lmax

        if isinstance(lmin, (bool, np.bool_)) or not isinstance(lmin, (int, np.integer)):
            raise TypeError("lmin must be an integer.")
        if isinstance(lmax, (bool, np.bool_)) or not isinstance(lmax, (int, np.integer)):
                    raise TypeError("lmax must be an integer.")
        if not 2 <= lmin <= lmax <= self.lmax:
            raise ValueError(
                f"Require 2 <= lmin <= lmax <= {self.lmax}."
            )

        num_lm = self.lmax * (self.lmax + 2) - 3
        idx_start = nindex(l=lmin, m=-lmin)
        idx_end = nindex(l=lmax, m=lmax)
        
        is_joint = (self.C_matrix.shape[0] == 2 * num_lm)
        matrix = self.norm_C_matrix if norm else self.C_matrix
        
        if is_joint:
            indices = np.concatenate([
                np.arange(idx_start, idx_end + 1),
                np.arange(idx_start, idx_end + 1) + num_lm
            ])
            return matrix[np.ix_(indices, indices)]
        else:
            return matrix[idx_start:idx_end+1, idx_start:idx_end+1]

    def compute_KL(self, s3_omk=None, lmin=None, lmax=None):
        """
        Compute the Kullback-Leibler (KL) divergence of correlation matrix compared to S3.
        Parameters
        ----------
        s3_omk : float or None, optional
            OmegaK of S3 matrix to be compared against.
            If None, set to self.OmegaK of lens space.
        lmin : int or None, optional
            min l to return matrix and calculate KL divergence.
            If None, set to 2.
        lmax : int or None, optional
            max l to return matrix and calculate KL divergence.
            If None, set to self.lmax.

        Returns
        -------
        KL: ndarray, shape (2,)
            Forward and backward divergences:
            [D_KL(topology || S3), D_KL(S3 || topology)].

        Notes
        -----
        Assumes zero-mean Gaussian fields. Both covariance matrices
        must be finite, Hermitian, and positive definite. Singular
        covariances are rejected; no regularization is applied.
        """
        if self.C_matrix is None or self.norm_C_matrix is None:
            raise RuntimeError("The correlation matrix has not been computed. Call compute_Clmlpmp() first.")
            
        
        if lmin is None: lmin = 2
        if lmax is None: lmax = self.lmax

        if not 2 <= lmin <= lmax <= self.lmax:
            raise ValueError(
                f"Require 2 <= lmin <= lmax <= {self.lmax}."
            )

        num_lm = self.lmax * (self.lmax + 2) - 3
        is_joint = (self.C_matrix.shape[0] == 2 * num_lm)

        s3_params = self.params.copy()
        if s3_omk is not None:
            s3_params['OmegaK'] = s3_omk
        s3_params['verbose'] = False
        s3_space = S3(s3_params)

        idx_start = nindex(l=lmin, m=-lmin)
        idx_end = nindex(l=lmax, m=lmax)
        
        sliced_C_matrix = self.get_C_matrix(norm=False, lmin=lmin, lmax=lmax)

        # compute C_l's for S3
        Cells = s3_space.get_manual_Cls()
        counts = 2 * np.arange(2, self.lmax + 1) + 1
        if not is_joint:
            tiled_s3 = np.repeat(Cells, counts)
            
            slice_TT = tiled_s3[idx_start:idx_end+1]
            S3_cov_sliced = np.diag(slice_TT)

            # KL_matrix =  sliced_C_matrix @ np.diag(1.0 / slice_TT)
            
        else:
            tiled_TT = np.repeat(Cells[0,:], counts)
            tiled_EE = np.repeat(Cells[2,:], counts)
            tiled_TE = np.repeat(Cells[1,:], counts)
            
            slice_TT = tiled_TT[idx_start:idx_end+1]
            slice_EE = tiled_EE[idx_start:idx_end+1]
            slice_TE = tiled_TE[idx_start:idx_end+1]
            
            S3_cov_sliced = np.block([
                [np.diag(slice_TT), np.diag(slice_TE)],
                [np.diag(slice_TE), np.diag(slice_EE)]
            ])

        # KL_matrix = np.linalg.solve(S3_cov_sliced,sliced_C_matrix)

        for label, covariance in (
            ('Lens-space', sliced_C_matrix),
            ('S3', S3_cov_sliced),
        ):
            if not np.all(np.isfinite(covariance)):
                raise ValueError(f"{label} covariance contains NaN or infinity.")

            scale = np.max(np.abs(covariance))
            if not np.allclose(
                covariance,
                covariance.conj().T,
                rtol=1e-10,
                atol=1e-12 * scale,
            ):
                raise ValueError(f"{label} covariance is not Hermitian.")
            
        from scipy.linalg import eigh

        lams = eigh(
            sliced_C_matrix,
            S3_cov_sliced,
            eigvals_only=True,
            check_finite=True,
        )

        if not np.all(np.isfinite(lams)):
            raise ValueError("The KL eigenvalues contain NaN or infinity.")

        if np.any(lams <= 0):
            raise ValueError(
                "KL requires positive-definite covariances. "
                f"Smallest generalized eigenvalue: {lams.min():.3e}."
            )
        
        forward_KL = 0
        backward_KL = 0
        
        for lam in lams:
            forward_KL += (lam - np.log(lam) - 1)
            backward_KL += (1.0/lam + np.log(lam) - 1)
            
        return np.array([forward_KL/2.0, backward_KL/2.0])

class S3(SphericalTopology):

    def __init__(self,params : dict = None) -> None:
        super().__init__(params)

    def get_Cls(self):
        """
        Compute CMB power spectrum using CAMB.
        """
        import camb 
        results = camb.get_results(self.cambpars)
        Cells = results.get_cmb_power_spectra(self.cambpars, raw_cl=True, lmax=self.lmax, 
                                            CMB_unit='muK')['unlensed_scalar']
        if self.onlyTT:
            return Cells[2:,0]
        else:
            Cells_TT = Cells[2:,0]
            Cells_EE = Cells[2:,1]
            Cells_TE = Cells[2:,3]
            return np.array([Cells_TT,Cells_TE,Cells_EE])
    
    def get_manual_Cls(self):
        """
        Compute CMB power spectrum manually.
        """
        power_spectrum = self.primpower()
        ps_factor = (power_spectrum/np.arange(3,self.nmax+2))
        transfer_funcs = self.transfer_functions()

        if self.onlyTT:
            transfer_squared = transfer_funcs**2
            Cells = 4*pi*transfer_squared @ ps_factor
            return Cells
        else:
            transfer_squared_T = transfer_funcs[0]**2
            transfer_squared_E = transfer_funcs[1]**2
            
            Cells_TT = 4*pi*transfer_squared_T @ ps_factor
            Cells_EE = 4*pi*transfer_squared_E @ ps_factor
            Cells_TE = 4*pi*transfer_funcs[0]*transfer_funcs[1] @ ps_factor
            
            return np.array([Cells_TT,Cells_TE,Cells_EE])

class EllipticSpace(SphericalTopology):

    def __init__(self,params : dict = None) -> None:
        super().__init__(params)

    def get_Cls(self):
        """
        Compute CMB power spectrum.
        """
        power_spectrum = self.primpower()
        ps_factor = (power_spectrum[0::2]/np.arange(3,self.nmax+2,2))
        transfer_funcs = self.transfer_functions()


        if self.onlyTT:
            transfer_squared = transfer_funcs**2
            Cells = 8*pi*transfer_squared[:,0::2] @ ps_factor
            return Cells
        else:
            transfer_squared_T = transfer_funcs[0]**2
            transfer_squared_E = transfer_funcs[1]**2
            
            Cells_TT = 8*pi*transfer_squared_T[:,0::2] @ ps_factor
            Cells_EE = 8*pi*transfer_squared_E[:,0::2] @ ps_factor
            Cells_TE = 8*pi*transfer_funcs[0][:,0::2]*transfer_funcs[1][:,0::2] @ ps_factor
            
            return np.array([Cells_TT,Cells_TE,Cells_EE])

    def compute_Clmlpmp(self):
        """
        Compute full CMB correlation matrix. 
        """
        power_spectrum = self.primpower()
        ps_factor = (power_spectrum[0::2]/np.arange(3,self.nmax+2,2))
        transfer_funcs = self.transfer_functions()

        counts = 2 * np.arange(2, self.lmax + 1) + 1

        if self.onlyTT:
            transfer_squared = transfer_funcs**2
            Cells = 8*pi*transfer_squared[:,0::2] @ ps_factor

            tiled_Cells = np.repeat(Cells, counts)

            C_final = np.block([np.diag(tiled_Cells)])

            Cells_s3 = 4*pi*transfer_squared @ (power_spectrum/np.arange(3,self.nmax+2))
            tiled_Cells_s3 = np.repeat(Cells_s3, counts)
            sqrtClClp = np.sqrt(np.outer(tiled_Cells_s3, tiled_Cells_s3))
            
        else:
            transfer_squared_T = transfer_funcs[0]**2
            transfer_squared_E = transfer_funcs[1]**2
            transfer_squared_TE = transfer_funcs[0]*transfer_funcs[1]
            
            Cells_TT = 8*pi*transfer_squared_T[:,0::2] @ ps_factor
            Cells_EE = 8*pi*transfer_squared_E[:,0::2] @ ps_factor
            Cells_TE = 8*pi*transfer_squared_TE[:,0::2] @ ps_factor

            tiled_TT = np.repeat(Cells_TT, counts)
            tiled_EE = np.repeat(Cells_EE, counts)
            tiled_TE = np.repeat(Cells_TE, counts)

            C_final = np.block([
                [np.diag(tiled_TT), np.diag(tiled_TE)],
                [np.diag(tiled_TE), np.diag(tiled_EE)]
                ])

            Cells_TT_s3 = 4*pi*transfer_squared_T @ (power_spectrum/np.arange(3,self.nmax+2))
            Cells_EE_s3 = 4*pi*transfer_squared_E @ (power_spectrum/np.arange(3,self.nmax+2))
            
            tiled_TT_s3 = np.repeat(Cells_TT_s3, counts)
            tiled_EE_s3 = np.repeat(Cells_EE_s3, counts)
            
            tiled_joint_s3 = np.concatenate([tiled_TT_s3, tiled_EE_s3])
            sqrtClClp = np.sqrt(np.outer(tiled_joint_s3, tiled_joint_s3))

        self.C_matrix = C_final
        self.norm_C_matrix = C_final/sqrtClClp

class LensSpace(SphericalTopology):

    def __init__(self,params : dict = None) -> None:
        super().__init__(params)

    def _compute_n_batch(self, n_list, num_lm, ell_arr, mm_arr,
                          transfer_funcs, power_spectrum, threads_per_worker,CHUNK_SIZE = 50000):
        """
        Main chunk of correlation matrix computation. 
        Calculates C_lmlpmp for specific n indices specified by n_list.
        Parameters
        ----------
        n_list : array_like
            List of n indices.
        num_lm : float
            Total number of (l,m) pairs for given lmax.
        ell_arr: array_like
            List of ell values for given n.
        mm_arr : array_like
            List of 2*m values for given n.
        transfer_funcs : ndarray
            Transfer function values.
        power_spectrum: ndarray
           Primordial power spectrum values.
        threads_per_worker : int
            Number of threads per worker.
        CHUNK_SIZE : int, optional
            Number of values to small-d matrices and CG coefficients per iteration.
            Computationally more efficient than calculating all values at once.
        """
        import traceback
        try:
            with threadpool_limits(limits=threads_per_worker):
                N_fields = 1 if self.onlyTT else 2
                C_local= np.zeros((N_fields*num_lm, N_fields*num_lm), dtype=np.complex128)

                for n in n_list:
                    
                    n_idx = n - 2
                    prefactor = power_spectrum[n_idx] / (n + 1)

                    pairs = find_mLmR_pairs(n, self.p, self.q)
                    if len(pairs) == 0: continue

                    mmL_all = (2 * pairs[:, 0]).astype(np.int32)
                    mmR_all = (2 * pairs[:, 1]).astype(np.int32)
                    num_pairs = len(mmL_all)

                    valid_ell_mask = (ell_arr >= 2) & (ell_arr <= n) & ((ell_arr - 2) < (transfer_funcs.shape[1] if not self.onlyTT else transfer_funcs.shape[0])) #transfer_funcs.shape[0] is lmax-1, this is saying ell<lmax+1
                    valid_i_global = np.where(valid_ell_mask)[0]
                    
                    mm_valid = mm_arr[valid_i_global]
                    ell_valid = ell_arr[valid_i_global]

                    if self.onlyTT:
                        delta_valid = np.ascontiguousarray(transfer_funcs[ell_valid - 2, n_idx])
                    else:
                        delta_valid_T = np.ascontiguousarray(transfer_funcs[0, ell_valid - 2, n_idx])
                        delta_valid_E = np.ascontiguousarray(transfer_funcs[1, ell_valid - 2, n_idx])

                    if abs(self.chi0 - pi/2) < 1e-8 or abs(self.chi0)<1e-8:
  
                        from scipy.sparse import coo_matrix

                        for k_start in range(0, num_pairs, CHUNK_SIZE):

                            k_end = min(k_start + CHUNK_SIZE, num_pairs)
                            mmL_list = mmL_all[k_start:k_end]
                            mmR_list = mmR_all[k_start:k_end]
                            chunk_pairs_len = len(mmL_list)

                            if abs(self.chi0 - pi/2) < 1e-8:
                                target_mm = mmL_list + mmR_list
                            if abs(self.chi0)<1e-8:
                                #Technically the wigner d also picks up a phase for chi0=0, 
                                # but diagonality in m would cancel it
                                target_mm = mmL_list - mmR_list
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

                            # The point of doing this weird thing is to do np.unique over a 1D array, 
                            # which is much faster than np.unique over a 2D aray and setting axis=0

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
                            
                            if self.onlyTT:
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
                        small_d = get_wigner_d_matrix_optimized(n,pi - 2 * self.chi0)

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

                            if self.onlyTT:
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
        
    def compute_Clmlpmp(self):
        """
        Compute full CMB correlation matrix. 
        """
        
        start = time()
        if self.verbose: print(f"Clmlpmp computation started. nmax is {self.nmax}")
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

            num_workers = max(1,int(num_workers//2))

        total_pool = get_available_cores()

        threads_per_worker = max(1, total_pool // num_workers)

        with threadpool_limits(limits=total_pool):
            power_spectrum = self.primpower()
            if self.onlyTT:
                transfer_funcs = self.transfer_functions()
                N_fields = 1
            else:
                transf_T, transf_E = self.transfer_functions()
                transfer_funcs = np.stack([transf_T, transf_E], axis=0)
                N_fields = 2

        lm_map = np.array([lmindex(i) for i in range(num_lm)])

        ell_arr = np.array([lm[0] for lm in lm_map], dtype=np.int32)
        m_arr = np.array([lm[1] for lm in lm_map], dtype=np.int32)
        mm_arr = 2 * m_arr

        if self.verbose: print(f"Distributing sum in n across {num_workers} parallel CPU cores...")
        
        worker_init = init_wigner_tables
        worker_initargs = (self.nmax * 2, 3)

        # Calculate n's sequentially starting from nmax, 
        # since this is the most computationally expensive
        all_ns_desc = list(range(self.nmax, 1, -1))
        num_modes = len(all_ns_desc)

        if self.batchsize is None:
            num_batch = min(num_workers, num_modes)
        else:
            num_batch = (num_modes + self.batchsize - 1) // self.batchsize
        
        # n_batches = [[] for _ in range(num_batch)] 
        n_batches = [all_ns_desc[i::num_batch]for i in range(num_batch)]
        
        # for i, n in enumerate(all_ns_desc):
        #     n_batches[i % num_batch].append(n)

        n_batches = [batch for batch in n_batches if len(batch) > 0]  

        C_total = np.zeros((N_fields * num_lm, N_fields * num_lm), dtype=np.complex128)

        try:
            # if q=1, lens space is homogeneous, so we can set chi0=0 before the calculation starts
            # save old chi0 value in a temp variable
            temp_chi = self.chi0
            if self.q==1: self.chi0=0

            # Calculate matrices
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=worker_init,
                initargs=worker_initargs
            ) as executor:
                
                futures = {
                    executor.submit(
                        self._compute_n_batch, 
                        batch, num_lm, ell_arr, mm_arr,
                        transfer_funcs, power_spectrum, threads_per_worker
                    )
                    for batch in n_batches
                }

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

                        futures.remove(future)
                        del local_C
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

                        futures.remove(future)
                        del local_C
        finally:
            # restore chi0 value if q=1
            self.chi0 = temp_chi

        # multiply by complex phase
        v=np.zeros(num_lm,dtype=np.complex128)
        for i, (ell,m) in enumerate(lm_map):
            phase = 1 if m%2==0 else -1
            v[i]=1j**(-ell)*sqrt(2*ell+1)*phase*np.exp(-1j*m*(self.xi_p0-self.xi_m0))
            
        phase_matrix=np.outer(v,np.conj(v))

        if not self.onlyTT:
            phase_matrix = np.block([
                [phase_matrix, phase_matrix],
                [phase_matrix, phase_matrix]
            ])

        free_wigner_tables()

        total_time = time() - start
        if self.verbose: print(f"Time taken: {total_time:.2f}s")

        C_final = self.p * 4 * pi * phase_matrix * C_total

        self.C_matrix = C_final

        # rather than initializing an S3 object, we'll calculate the normalization
        # using the transfer functions and power spectrum that are already calculated and stored
        # this saves about 3 secs of computation time
        counts = 2 * np.arange(2, self.lmax + 1) + 1
        if self.onlyTT:
            transfer_squared = transfer_funcs**2
            Cells = 4*pi*transfer_squared @ (power_spectrum/np.arange(3,self.nmax+2))
            tiled_Cells = np.repeat(Cells, counts)
            sqrtClClp = np.sqrt(np.outer(tiled_Cells, tiled_Cells))
        else:
            transfer_squared_T = transfer_funcs[0]**2
            transfer_squared_E = transfer_funcs[1]**2
            
            Cells_TT = 4*pi*transfer_squared_T @ (power_spectrum/np.arange(3,self.nmax+2))
            Cells_EE = 4*pi*transfer_squared_E @ (power_spectrum/np.arange(3,self.nmax+2))
            
            tiled_TT = np.repeat(Cells_TT, counts)
            tiled_EE = np.repeat(Cells_EE, counts)
            
            tiled_joint = np.concatenate([tiled_TT, tiled_EE])
            sqrtClClp = np.sqrt(np.outer(tiled_joint, tiled_joint))

        self.norm_C_matrix = C_final/sqrtClClp

    def plot_Clmlpmp(self, filename=None, m_ordering=False, positive_m_only=False,cbar_min_exp=-8,
                     usetex=False,show=True):
        """
        Plot correlation matrix normalized to sqrt(Cl*Clp) from S3.
        Parameters
        ----------
        filename : str or None, optional
            Path name to save plot as figure.
            If None, file is not saved.
        m_ordering : bool, optional
            If True, Plot matrix in (m,l) ordering.
            If False, matrix is plot in (l,m) ordering.
        positive_m_only : bool, optional
            If m_ordering is True, option to plot only non-negative m-values.
        cbar_min_exp : float, optional
            Minimum value of log10(abs(self.norm_C_matrix)) of colorbar.
        usetex : bool, optional
            Use LaTeX formatting in figure text.
        show : bool, optional
            Show plot in output.
        """
        from matplotlib import pyplot as plt 
        import matplotlib.ticker as ticker

        # set figure style
        style = {
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif'],
            'mathtext.fontset': 'cm',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            
            'axes.linewidth': 1.2,          
            'xtick.direction': 'in',        
            'ytick.direction': 'in',
            'xtick.top': True,              
            'ytick.right': True,            
            'xtick.minor.visible': False,    
            'ytick.minor.visible': False,
            'xtick.major.size': 6,          
            'xtick.minor.size': 3,          
            'ytick.major.size': 6,
            'ytick.minor.size': 3,
            'xtick.major.width': 1.0,       
            'ytick.major.width': 1.0,
            
            'lines.linewidth': 1.5,         
            'legend.frameon': True,        
            'legend.loc': 'best',
            
            'figure.figsize': (5.0, 5.0),   
            'figure.dpi': 150,              
            'savefig.bbox': 'tight',        
            'savefig.pad_inches': 0.1
        }

        style['text.usetex'] = usetex
        if usetex:
            style.update({
                'font.family': 'serif',
                'font.serif': ['Computer Modern Roman'],
                'text.latex.preamble': r'\usepackage{amsmath, amssymb}',
                })

        with plt.rc_context(rc=style):
            if self.C_matrix is None or self.norm_C_matrix is None:
                 raise RuntimeError("The correlation matrix has not been computed. Call compute_Clmlpmp() first.")
    
            num_lm = self.lmax * (self.lmax + 2) - 3
            is_joint = (self.C_matrix.shape[0] == 2 * num_lm)
            
            matrix_to_plot = self.norm_C_matrix.copy()

            if m_ordering:
                m_idx = get_m_ordering_indices(self.lmax, lmin=2, positive_m_only=positive_m_only)
                
                if is_joint:
                    all_idx = np.concatenate([m_idx, m_idx + num_lm])
                    matrix_to_plot = matrix_to_plot[np.ix_(all_idx, all_idx)]
                else:
                    matrix_to_plot = matrix_to_plot[np.ix_(m_idx, m_idx)]
                    
                start_m = 0 if positive_m_only else -self.lmax
                ms = np.arange(start_m, self.lmax + 1)

                counts = np.array([self.lmax - max(2, abs(m)) + 1 for m in ms])
                
                valid_ms = counts > 0
                block_labels = ms[valid_ms]
                counts = counts[valid_ms]
                
                str_labels = [f'${m}$' for m in block_labels]
                xlabel_str = r'$m$'
                ylabel_str = r'$m^\prime$'
            else:
                ells = np.arange(2, self.lmax + 1)
                counts = 2 * ells + 1
                
                str_labels = [f'${ell}$' for ell in ells]
                xlabel_str = r'$\ell$'
                ylabel_str = r'$\ell^\prime$'

            fig = plt.figure(figsize=(8,8) if is_joint else (6,6))
        
            cmap = plt.cm.inferno.copy()
            cmap.set_bad(color='black')
            with np.errstate(divide="ignore"):
                log_matrix = np.log10(np.abs(matrix_to_plot))
            plt.imshow(log_matrix, cmap=cmap, vmin=cbar_min_exp, origin='lower')

            block_ends = np.cumsum(counts)
            start_indices = np.concatenate(([0], block_ends[:-1]))
            tick_positions = start_indices + counts / 2.0 - 0.5

            boundaries = block_ends - 0.5
            internal_boundaries = boundaries[:-1]
            N = boundaries[-1] + 0.5 
            
            ax = fig.gca()

            if is_joint:
                field_size = len(matrix_to_plot) // 2
                
                all_tick_positions = np.concatenate([tick_positions, tick_positions + field_size])
                tick_labels_joint = str_labels * 2

                all_boundaries = np.concatenate([
                    internal_boundaries, 
                    [N - 0.5], 
                    internal_boundaries + field_size
                ])
                
                plt.vlines(all_boundaries, ymin=-0.5, ymax=2*N-0.5, colors='white', linewidth=0.3, alpha=0.3)
                plt.hlines(all_boundaries, xmin=-0.5, xmax=2*N-0.5, colors='white', linewidth=0.3, alpha=0.3)

                plt.axvline(x=field_size - 0.5, color='white', linewidth=0.8, alpha=0.8)
                plt.axhline(y=field_size - 0.5, color='white', linewidth=0.8, alpha=0.8)
                
                ax.set_xticks(all_tick_positions)
                ax.set_xticklabels(tick_labels_joint, rotation=0)
                ax.set_yticks(all_tick_positions)
                ax.set_yticklabels(tick_labels_joint)

                ax.annotate("$T$", xy=(0.25, -0.05), xycoords='axes fraction', fontsize=16, ha='center', va='top', annotation_clip=False)
                ax.annotate("$E$", xy=(0.75, -0.05), xycoords='axes fraction', fontsize=16, ha='center', va='top', annotation_clip=False)

                ax.annotate("$T$", xy=(-0.05, 0.25), xycoords='axes fraction', fontsize=16, ha='right', va='center', annotation_clip=False)
                ax.annotate("$E$", xy=(-0.05, 0.75), xycoords='axes fraction', fontsize=16, ha='right', va='center', annotation_clip=False)
                
            else:
                plt.vlines(internal_boundaries, ymin=-0.5, ymax=N-0.5, colors='white', linewidth=0.5, alpha=0.5)
                plt.hlines(internal_boundaries, xmin=-0.5, xmax=N-0.5, colors='white', linewidth=0.5, alpha=0.5)
                
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(str_labels, rotation=0)
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(str_labels)

            label_pad = 30 if is_joint else 10
            ax.set_xlabel(xlabel_str, fontsize=14, labelpad=label_pad)
            ax.set_ylabel(ylabel_str, fontsize=14, labelpad=label_pad, rotation=0)

            # Colorbar formatting
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: rf'$10^{{{x:g}}}$'))
            cbar.ax.set_title(r'$\left|\frac{C_{\ell m \ell^\prime m^\prime}}{\sqrt{C_\ell C_{\ell^\prime}}}\right|$', pad=20)

            # Title formatting in a single line
            title_str = (
            f'$L({self.p},{self.q})$ with ' 
            + r'$\Omega_{\rm K}=$' + f' ${np.round(self.OmegaK,4)}$ and ' 
            + r'$(\chi_0,{\xi_+}_0,{\xi_-}_0)=$' + f' $({np.round(self.chi0,2)},{np.round(self.xi_p0,2)},{np.round(self.xi_m0,2)})$'
        )
                
            ax.set_title(title_str, fontsize=12)

            fig.tight_layout()
            
            if filename is not None:
                plt.savefig(filename,bbox_inches='tight',dpi=300)

            if show: plt.show()
            else: plt.close()

            return (fig,ax)