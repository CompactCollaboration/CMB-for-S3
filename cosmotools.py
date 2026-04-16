import numpy as np
import scipy.integrate as integ
from scipy.constants import c
import math
from s3tools import omk2R, R2omk

rng = np.random.default_rng()



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

def Einv(z, Om, OK, OL):
    zp1 = 1 + z
    return 1 / np.sqrt(Om * zp1**3 + OK * zp1**2 + OL)

def chi(z, Om, OK, OL):
    return np.sqrt(np.abs(-OK)) * integ.quad(Einv, 0, z, args=(Om, OK, OL))[0]

def find_d_lss(OK,H0=67.5):
    """
    Calculates distance to last scattering surface for a given OmegaK. Units are in Mpc.
    """
    c_H0 = c / H0
    zLS = 1090
    Om = 0.314 - 3.71 * OK
    OL = 1 - Om - OK

    return 2 * c_H0 * integ.quad_vec(Einv, 0, zLS, args=(Om, OK, OL))[0]/1000

def is_allowed_p(p,OK):
    """
    Determine if a given p is allowed for a given OmegaK. Based on Eq. (5.3) in Paper Ic.
    Returns True or False.
    """
    alpha = 0.761
    dLSS = find_d_lss(OK)
    pstar = (2 * np.pi * alpha / dLSS)**2
    if p <= pstar:
        return True
    else:
        return False