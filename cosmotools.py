import numpy as np
from scipy.integrate import quad,quad_vec
from scipy.constants import c
from math import gcd

def is_homeomorphic(p, q1, q2):
    """
    Test the homeomorphic condition for Lens spaces L(p, q1) and L(p, q2).
    Parameters
    ----------
    p : int
    q1 : int
    q2 : int
    """
    return (q1 == q2%p) or (q1 == (-q2%p)) or ((q1*q2)%p == 1) \
            or ((q1*q2)%p == (-1)%p)

def get_unique(p):
    """
    Find q's for which L(p,q) is not homeomorphic with other q's in the returned list.
    Parameters
    ----------
    p : int
    """
    qlist = []
    for q2 in range(2, p//2+1):
        # p and q2 must be relatively prime
        if gcd(p, q2) != 1:
            continue
        for q1 in qlist:
            if is_homeomorphic(p, q1, q2):
                break
        else: # One of the oddest features in Python!
            qlist.append(q2)
    return qlist

def Einv(z, OmegaM, OmegaK, OmegaL, OmegaR=0.0):
    """
    Calculate 1/E(z) for given cosmology.
    Parameters
    ----------
    z : float
        Redshift.
    OmegaM : float
        Matter density parameter today.
    OmegaK : float
        Curvature density parameter today.
    OmegaL : float
        Dark energy density parameter today.
    OmegaR : float, optional
        Radiation density parameter today. Default is 0.
    """
    zp1 = 1 + z
    return 1 / np.sqrt(OmegaR * zp1**4 + OmegaM * zp1**3 + OmegaK * zp1**2 + OmegaL)

def chi(z, OmegaM, OmegaK, OmegaL, OmegaR=0.0, normalize=True, H0=67.5):
    """
    Calculate comoving distance for given cosmology.
    Parameters
    ----------
    z : float
        Redshift.
    OmegaM : float
        Matter density parameter today.
    OmegaK : float
        Curvature density parameter today.
    OmegaL : float
        Dark energy density parameter today.
    OmegaR : float, optional
        Radiation density parameter today. Default is 0.
    normalize : bool, optional
        If True, returned in units of the curvature radius Rc. Otherwise, units are Mpc.
    H0 : float, optional
        Hubble parameter in units km/s/Mpc. Default is 67.5.
    """
    if normalize == False:
        c_H0 = (c/1000) / H0 # in Mpc
        return c_H0 * quad(Einv, 0, z, args=(OmegaM, OmegaK, OmegaL, OmegaR))[0]
    else:
        return np.sqrt(np.abs(-OmegaK)) * quad(Einv, 0, z, args=(OmegaM, OmegaK, OmegaL, OmegaR))[0]

def find_d_lss(OmegaK,H0=67.5,normalize=False):
    """
    Calculates diameter of the last scattering surface for a given OmegaK.
    Parameters
    ----------
    OmegaK : float
        Curvature density parameter today.
    H0 : float, optional
        Hubble parameter in units km/s/Mpc. Default is 67.5.
    normalize : bool, optional
        If False, returns value in units of Mpc. Else, d_lss is normalized to the curvature radius Rc.
    """
    zLS = 1090
    OmegaR = 4.176e-5/(H0/100.)**2
    OmegaM = (0.022 + 0.122 + 0.06/93.14) / (H0/100.)**2 # This matches the CAMB initialization
    OmegaL = 1 - OmegaM - OmegaK - OmegaR

    if normalize:
        return  2 * np.sqrt(np.abs(OmegaK)) * quad_vec(Einv, 0, zLS, args=(OmegaM, OmegaK, OmegaL, OmegaR))[0]
    else:
        c_H0 = (c/1000) / H0 # in Mpc
        return 2 * c_H0 * quad_vec(Einv, 0, zLS, args=(OmegaM, OmegaK, OmegaL, OmegaR))[0]

def find_d_nc(p,OmegaK,q=1,chi0=0,H0=67.5,normalize=True):
    """
    Calculates the distance to the nearest clone for a given lens space.
    Parameters
    ----------
    p : int
        Integer specifying lens space L(p,q).
    OmegaK : float
        Curvature density parameter today.
    q : int, optional
        Integer specifying lens space L(p,q). Default is 1.
    chi0 : float, optional
        Observer location in toroidal coordinates with respect to main axis.
        Must be between 0 and pi/2. Default is 0.
    H0 : float, optional
        Hubble parameter in units km/s/Mpc. Default is 67.5.
    normalize : bool, optional
        If False, returns value in units of Mpc. Else, d_nc is normalized to d_lss.
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
        d_lss = find_d_lss(OmegaK,H0=H0)
        return d_nc/d_lss
        
    else: return d_nc

def get_omk_from_clone_distance(p,ratio,q=1,chi0=0,H0=67.5,tol=0.005,max_iter=100):
    """
    Given a lens space L(p,q), find what OmegaK will give the desired ratio of d_nc/d_lss using bisection method.
    Parameters
    ----------
    p : int
        Integer specifying lens space L(p,q).
    ratio : float
        Ratio of d_nc/d_lss.
    q : int, optional
        Integer specifying lens space L(p,q). Default is 1.
    H0 : float, optional
        Hubble parameter in units km/s/Mpc. Default is 67.5.
    tol : float, optional
        Maximum tolerance between d_nc/d_lss ratio for the calculated OmegaK and the specified ratio.
        Default is 0.005, meaning the calculated ratio will be within 99.5% accuracy of the specified ratio.
    max_iter : int, optional
        Max iterations for bisection method.
    """

    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError(
            "ratio must be finite and greater than zero."
        )

    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError(
            "tol must be finite and between zero and one."
        )

    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError("max_iter must be a positive integer.")

    def evaluate(omk):
        value = find_d_nc(p, omk, q=q, chi0=chi0,H0=H0, normalize=True)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("The clone-distance calculation returned an invalid ratio.")
        return value

    def within_tolerance(value):
        # Preserve your existing symmetric relative-error criterion.
        error = abs(value - ratio) / (0.5 * (value + ratio))
        return error <= tol

    lo, hi = -0.1, -1e-6
    r_lo, r_hi = evaluate(lo), evaluate(hi)
    r_min, r_max = sorted((r_lo, r_hi))

    if not r_min <= ratio <= r_max:
        raise ValueError(f"Requested ratio {ratio:g} is outside [{r_min:g}, {r_max:g}] "
                        f"for OmegaK in [{lo:g}, {hi:g}].")

    if within_tolerance(r_lo):
        return lo
    if within_tolerance(r_hi):
        return hi

    f_lo = r_lo - ratio

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = evaluate(mid)

        if within_tolerance(value):
            return mid

        f_mid = value - ratio

        # Replace the endpoint with the same residual sign.
        if (f_lo < 0) == (f_mid < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid

    raise RuntimeError(f"Curvature search did not converge after {max_iter} iterations.")