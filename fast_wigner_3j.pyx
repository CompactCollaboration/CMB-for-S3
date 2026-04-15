# fast_wigner_3j.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np

# Declare the C functions from the wigxjpf header
cdef extern from "wigxjpf.h":
    void wig_table_init(int max_two_j, int wigner_type) nogil
    void wig_temp_init(int max_two_j) nogil
    void wig_table_free() nogil
    void wig_temp_free() nogil
    double wig3jj(int two_j1, int two_j2, int two_j3, int two_m1, int two_m2, int two_m3) nogil

# Expose initialization functions to Python
def init_wigner_tables(int max_two_j, int wigner_type=3):
    wig_table_init(max_two_j, wigner_type)
    wig_temp_init(max_two_j)

def free_wigner_tables():
    wig_table_free()
    wig_temp_free()

# Fast vectorized calculation
def compute_wig3j_flat(int n, int[:] ell_arr_flat, int[:] mm_arr_flat, int[:] mmL_arr_flat):
    cdef int num_vals = ell_arr_flat.shape[0]
    
    # Pre-allocate the numpy array and get a fast C-level memoryview
    cdef np.ndarray[np.float64_t, ndim=1] w3j_vals = np.empty(num_vals, dtype=np.float64)
    cdef double[:] w3j_vals_view = w3j_vals
    
    cdef int i, j3, m3, m1, m2
    
    # RELEASE THE GIL! This loop now runs at pure C speed
    with nogil:
        for i in range(num_vals):
            # wigxjpf expects all inputs as 2*j and 2*m
            j3 = 2 * ell_arr_flat[i]
            
            # Since mm_arr_flat and mmL_arr_flat are already 2*m (from Python), 
            # we just need to handle the minus sign for m3 and standard math for m2.
            m3 = -mm_arr_flat[i]
            m1 = mmL_arr_flat[i]
            m2 = mm_arr_flat[i] - mmL_arr_flat[i]
            
            w3j_vals_view[i] = wig3jj(n, n, j3, m1, m2, m3)
            
    return w3j_vals