---
title: GPU-accelerated Libraries
---

## GPU-accelerated Libraries

### CUDA Libraries

* There are GPU-accelerated libraries available for:
    - random number generation
    - fast Fourier transforms
    - BLAS/LAPACK
    - sparse linear algebra
* Nvidia ships cuRAND, cuFFT and cuBLAS with the CUDA toolkit
    - 3rd party alternatives including MAGMA

### Converting existing code

* Each library contains pre-written GPU kernels implementing common operations optimised for several classes of GPU
* All that is required is to convert existing code that calls a CPU library to follow the upload/execute/synchronise/download pattern
    - allocate GPU memory and upload data using the CUDA runtime library
    - execute pre-written kernel from the specific CUDA library required
    - synchronize, download results and free GPU memory using the CUDA runtime library

### SAXPY from cuBLAS

### SGEMM from cuBLAS
