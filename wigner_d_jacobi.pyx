# wigner_d_jacobi.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

cimport cython
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, log, exp, lgamma, fabs, ceil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_wigner_d_matrix_optimized(int j_times_2, double beta):
    """
    Computes the Wigner d-matrix for j = j_times_2 / 2 using log-scaled 
    Jacobi polynomials to prevent underflow at extreme n values (e.g., 3500).
    """
    cdef int size = j_times_2 + 1
    cdef double j = j_times_2 / 2.0
    
    # Allocate the output matrix
    cdef double[:, ::1] d_mat = np.zeros((size, size), dtype=np.float64)

    # Precompute trigonometry and logs
    cdef double half_beta = 0.5 * beta
    cdef double sin_hb = sin(half_beta)
    cdef double cos_hb = cos(half_beta)

    cdef double log_sin = -1.0e30
    cdef double log_cos = -1.0e30
    if sin_hb > 1e-15:
        log_sin = log(sin_hb)
    if cos_hb > 1e-15:
        log_cos = log(cos_hb)

    cdef double cos_b = cos(beta)

    # Rescaling constants
    cdef double HUGE_VAL = 1.0e30
    cdef double TINY_VAL = 1.0e-30
    cdef double LOG_HUGE_VAL = log(HUGE_VAL)

    cdef int start_r = <int>ceil(j)
    
    # Thread-local variables for the parallel loop
    cdef int r, c, c_start, c_end, n
    cdef double mp, m, k, a, b, log_k_fact
    cdef double log_A, log_ang, prefactor, log_rescale, total_log
    cdef double poly, p_nm1, p_nm2, n_f, ab_n, ab_2n, c1, c2, c3
    cdef double val, sign

    # Parallelize over the rows using OpenMP
    for r in prange(start_r, size, nogil=True, schedule='dynamic'):
        mp = r - j
        k = j - mp
        
        if k < 0:
            continue

        log_k_fact = lgamma(k + 1.0)
        c_start = <int>(2.0 * j - r)
        c_end = r + 1

        for c in range(c_start, c_end):
            m = c - j
            a = mp - m
            b = mp + m

            # Compute prefactor entirely in log-space to survive massive factorials
            log_A = 0.5 * (
                log_k_fact
                + lgamma(k + a + b + 1.0)
                - lgamma(k + a + 1.0)
                - lgamma(k + b + 1.0)
            )

            log_ang = (a * log_sin) + (b * log_cos)
            log_rescale = 0.0

            # 3-term recurrence for Jacobi polynomials
            if k == 0:
                poly = 1.0
            elif k == 1:
                poly = 0.5 * ((a + b + 2.0) * cos_b + (a - b))
            else:
                p_nm2 = 1.0
                p_nm1 = 0.5 * ((a + b + 2.0) * cos_b + (a - b))
                poly = 0.0

                for n in range(2, <int>k + 1):
                    n_f = <double>n
                    ab_n = a + b + n_f
                    ab_2n = a + b + 2.0 * n_f

                    c1 = 2.0 * n_f * (ab_n) * (ab_2n - 2.0)
                    c2 = (ab_2n - 1.0) * (ab_2n * (ab_2n - 2.0) * cos_b + (a * a - b * b))
                    c3 = 2.0 * (n_f + a - 1.0) * (n_f + b - 1.0) * ab_2n

                    poly = (c2 * p_nm1 - c3 * p_nm2) / c1
                    p_nm2 = p_nm1
                    p_nm1 = poly

                    # Dynamic rescaling to prevent float64 explosion
                    # FIXED: Removed inplace operators (*=, +=) so Cython doesn't treat them as reductions
                    if fabs(poly) > HUGE_VAL:
                        poly = poly * TINY_VAL
                        p_nm1 = p_nm1 * TINY_VAL
                        p_nm2 = p_nm2 * TINY_VAL
                        log_rescale = log_rescale + LOG_HUGE_VAL

            # Recombine
            total_log = log_A + log_ang + log_rescale

            if total_log < -100.0:
                val = 0.0
            else:
                if m >= mp:
                    sign = 1.0
                else:
                    sign = -1.0 if (<int>a % 2 != 0) else 1.0
                val = sign * exp(total_log) * poly

            # Populate current quadrant
            d_mat[r, c] = val

            # Populate symmetry quadrants
            sign = 1.0 if ((r - c) % 2 == 0) else -1.0
            d_mat[c, r] = sign * val
            d_mat[size - 1 - r, size - 1 - c] = sign * val
            d_mat[size - 1 - c, size - 1 - r] = val

    return np.asarray(d_mat)