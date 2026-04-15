from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

extra_compile_args = ["-O3", "-march=native", "-fopenmp"]
extra_link_args = ["-fopenmp"]
math_lib = ["m"]  

if os.name == "nt": 
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []
    math_lib = []  

# Macro to silence the NumPy deprecation warnings
numpy_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

# --- 1. Wigner d-matrix Extension ---
ext_wigner_d = Extension(
    name="wigner_d_jacobi",
    sources=["wigner_d_jacobi.pyx"],
    include_dirs=[np.get_include()],
    libraries=math_lib,
    define_macros=numpy_macros,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

# --- 2. Fast Wigner 3j Extension ---
ext_wigner_3j = Extension(
    name="fast_wigner",
    sources=["fast_wigner_3j.pyx"],
    include_dirs=[np.get_include(), "."], 
    libraries=math_lib + ["wigxjpf"],
    library_dirs=["."],
    runtime_library_dirs=["."] if os.name != "nt" else [],
    define_macros=numpy_macros,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="spherical_topology_cython_modules",
    ext_modules=cythonize(
        [ext_wigner_d, ext_wigner_3j], 
        language_level=3
    )
)