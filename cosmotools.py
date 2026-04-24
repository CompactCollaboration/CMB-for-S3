import numpy as np
import scipy.integrate as integ
from scipy.constants import c
import math
import pandas as pd
import pathlib
from s3tools import omk2R, R2omk

def is_homeomorphic(p, q1, q2):
    """
    Test the homeomorphic condition for Lens spaces L(p, q1) and L(p, q2).
    Returns True or False.
    """
    return (q1 == q2%p) or (q1 == (-q2%p)) or ((q1*q2)%p == 1) \
            or ((q1*q2)%p == (-1)%p)

def get_unique(p):
    qlist = []
    for q2 in range(2, p//2+1):
        # p and q2 must be relatively prime
        if math.gcd(p, q2) != 1:
            continue
        for q1 in qlist:
            if is_homeomorphic(p, q1, q2):
                break
        else: # One of the oddest features in Python!
            qlist.append(q2)
    return qlist

def get_unique_fast(p):
    """
    O(N) generation of unique q parameters for Lens space L(p, q).
    Uses modular multiplicative inverse to avoid nested loops.
    """
    seen = set()
    unique_q = []
    
    for q in range(2, p // 2 + 1):
        # p and q2 must be relatively prime
        if math.gcd(p, q) != 1:
            continue
        if q not in seen:
            unique_q.append(q)
            # Add all homeomorphic equivalents to the seen set
            inv_q = pow(q, -1, p)
            seen.update((
                q, 
                (-q) % p, 
                inv_q, 
                (-inv_q) % p
            ))
    return unique_q


def clone_separations(p, q, s):
    """
    Calculate all clone separations for L(p,q) given the parameter s.
    Inputs:
      p, q: integers: parameters of the lens space L(p,q).
      s: 1d array of size N: parameter determining the location of points in S^3
          This must be an array and the values must be in [0, 1].
    Outputs:
      distances: N x p-1 array: 2d array of clone distances.
             Stored in the order shown: 
                 First axis is the parameter number from the original array.
                 Second axis is the clone number, there are p-1 clones.     
    """
    cos = np.cos(2*np.pi * np.arange(1, p) / p)
    cosq = np.cos(2*np.pi * np.arange(1, p) * q / p)
    cosangs = s[:, np.newaxis] * cos[np.newaxis, :] + (1 - s)[:, np.newaxis] * cosq[np.newaxis, :]
    return np.arccos(cosangs)


def do_dmax_iteration(p, q, smin, smax, Nsteps=1000):
    s = np.linspace(smin, smax, Nsteps)
    dmin = clone_separations(p, q, s).min(axis=-1)
    dmax = dmin.max()
    indmax = dmin.argmax()
    return s[indmax-1], s[indmax+1], dmax

def find_dmax(p, q, maxiter=10, tol=1e-7, Nsteps=100):
    dmax = -1
    dmaxprev = 1
    smin, smax = 0.0, 1.0
    niter = 0
    while niter <= maxiter and not np.allclose(dmax, dmaxprev, rtol=tol, atol=tol):
        dmaxprev = dmax
        smin, smax, dmax = do_dmax_iteration(p, q, smin, smax, Nsteps=Nsteps)
        niter += 1
    return dmax, niter

def find_dmax_fast(p, q, maxiter=10, tol=1e-7, Nsteps=100):
    """
    Optimized distance finder. 
    Precomputes trig functions and operates entirely on cosines to avoid np.arccos in the loop.
    """
    
    # Precompute cos and cosq once per (p, q)
    cos = np.cos(2 * np.pi * np.arange(1, p) / p)
    cosq = np.cos(2 * np.pi * np.arange(1, p) * q / p)

    smin, smax = 0.0, 1.0
    niter = 0
    
    # We track the minimum of the maximum cosines
    min_max_cos = 2.0 
    min_max_cos_prev = -2.0
    
    while niter <= maxiter and not np.allclose(min_max_cos, min_max_cos_prev, rtol=tol, atol=tol):
        min_max_cos_prev = min_max_cos
        
        s = np.linspace(smin, smax, Nsteps)
        
        # Calculate cosangs using broadcasting (Shape: Nsteps x p-1)
        cosangs = s[:, np.newaxis] * cos + (1 - s)[:, np.newaxis] * cosq
        
        # Maximize the cosine (equivalent to minimizing the distance/arccos) across clones
        max_cos_per_s = cosangs.max(axis=-1)
        
        # Minimize the maximum cosine across the s parameter steps
        indmin = max_cos_per_s.argmin()
        min_max_cos = max_cos_per_s[indmin]
        
        # Safely determine the new bounds for the zoom-in search
        idx_left = max(0, indmin - 1)
        idx_right = min(Nsteps - 1, indmin + 1)
        
        smin, smax = s[idx_left], s[idx_right]
        niter += 1
        
    # Apply arccos only once at the very end
    dmax = np.arccos(min_max_cos)
    
    return dmax, niter

def get_all_lens_distances(pmax):
    lens_distance_file2 = pathlib.Path(f'lens_distances_{pmax}_pq.hdf5')
        
    if lens_distance_file2.exists():
        df2 = pd.read_hdf(lens_distance_file2, 'clonedistances')
        print("Loaded existing data.")
    else:
        res = []
        nitermax = -1

        print(f'Finding lens distances up to pmax = {pmax}. This may take a few minutes...')
        for p in range(5, pmax + 1):

            for q in get_unique_fast(p):
                dmax, niter = find_dmax_fast(p, q, Nsteps=100)

                if niter > nitermax:
                    nitermax = niter
                if niter >= 10:
                    print(f"Failed for {p}, {q}")

                res.append({'p': p, 'q': q, 'dmax': dmax})
            
            ratio = p/pmax
            if ratio*100 % 10 == 0: print(f'Computed p {p}/{pmax}. {ratio*100:.1f}% completed.')
        
        # 3. Save results
        df2 = pd.DataFrame.from_dict(res)
        df2.to_hdf(lens_distance_file2, key='clonedistances')
        print("Computation complete and saved.")

    return df2

def Einv(z, OmegaM, OmegaK, OmegaL):
    zp1 = 1 + z
    return 1 / np.sqrt(OmegaM * zp1**3 + OmegaK * zp1**2 + OmegaL)

def chi(z, OmegaM, OmegaK, OmegaL):
    return np.sqrt(np.abs(-OmegaK)) * integ.quad(Einv, 0, z, args=(OmegaM, OmegaK, OmegaL))[0]

def find_d_lss(OmegaK,H0=67.5,normalize=False):
    """
    Calculates distance to last scattering surface for a given OmegaK.
    If normalize=False, units are in Mpc. This is the default. Else, d_lss is normalized to Rc
    """
    zLS = 1090
    OmegaM = 0.314 - 3.71 * OmegaK
    OmegaL = 1 - OmegaM - OmegaK

    if normalize:
        return  2 * np.sqrt(np.abs(OmegaK)) * integ.quad_vec(Einv, 0, zLS, args=(OmegaM, OmegaK, OmegaL))[0]
    else:
        c_H0 = (c/1000) / H0 # in Mpc
        return 2 * c_H0 * integ.quad_vec(Einv, 0, zLS, args=(OmegaM, OmegaK, OmegaL))[0]

def find_d_nc(p,OmegaK,q=1,chi0=0,H0=67.5,normalize=True):
    """
    Calculates the distance to the nearest clone for a given OmegaK and p.
    If normalize=True, result is normalized to d_LSS. Otherwise, returned result is in Mpc.
    """
    c_H0 = (c/1000) / H0 # in Mpc
    Rc = c_H0/np.sqrt(np.abs(OmegaK))

    if q == 1:
        d_nc = 2*np.pi*Rc / p

    else:
        d = np.zeros(p-1)
        s = np.cos(chi0)**2

        for j in range(1,p):
            
            d[j-1] = np.arccos(s*np.cos(2*np.pi*j/p)+(1-s)*np.cos(2*np.pi*j*q/p))
        d_nc = np.min(d)*Rc

    if normalize:
        d_lss = find_d_lss(OmegaK)
        return d_nc/d_lss
        
    else: return d_nc

def get_omk_from_clone_distance(p,ratio,q=1,chi0=0,tol=0.005,H0=67.5):
    """
    Given a lens space L(p,1), find what OmegaK will give the desired ratio of d_NC/d_LSS
    """

    omk1 = -1e-6
    omk2 = -1e-1
    omk_guess = np.mean([omk1,omk2])
    guess_ratio = find_d_nc(p,omk_guess,normalize=True,q=q,chi0=chi0,H0=H0)
    limit = abs(guess_ratio-ratio)/np.mean([guess_ratio,ratio])
    condition = (limit <= tol)

    while not condition:
        # print(omk1,omk2,omk_guess,guess_ratio,flush=True)
        if guess_ratio < ratio:
            omk2 = omk_guess
        else:
            omk1 = omk_guess
        
        omk_guess = np.mean([omk1,omk2])
        guess_ratio = find_d_nc(p,omk_guess,normalize=True,q=q,chi0=chi0,H0=H0)
        limit = abs(guess_ratio-ratio)/np.mean([guess_ratio,ratio])
        condition = (limit <= tol)
    
    # print(guess_ratio,ratio,limit,tol)

    return omk_guess

'''
def get_p_from_clone_distance(OmegaK,ratio,H0=67.5):
    """
    Given an OmegaK, find what lens space L(p,1) will give the desired ratio of d_NC/d_LSS
    """
    c_H0 = (c/1000) / H0 # in Mpc

    d_lss = find_d_lss(OmegaK)
    d_nc = ratio*d_lss
    return int(np.round(2*np.pi*c_H0 / (d_nc * np.sqrt(np.abs(OmegaK)))))

def find_upper_p(OmegaK):
    """
    Determines the upper limit on p for a given OmegaK. Based on Eq. (5.3) in Paper Ic.
    """
    alpha = 0.761
    dLSS = find_d_lss(OmegaK,normalize=True) # want dLSS in units of Rc

    return (2 * np.pi * alpha / dLSS)**2
'''