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

* The following code snippet replaces the ```saxpy_fast``` function with the equivalent ```cublasSaxpy```:
{{cppfrag('09','saxpy/cublas.c','cublas_saxpy')}}

```
n = 10000, incx = 1, incy = 1
saxpy: 0.010205ms
saxpy_fast: 0.002530ms
```

### Why isn't it faster?

* GPUs have about 100x the computing power of a CPU
    - but only 20x the memory bandwidth
* SAXPY performs one floating point operation for each element in memory
    - the performance is bound by the memory bandwidth
* Matrix Multiply (SGEMM), however, performs ```2k``` operations per element
    - matrix multiply: ```C = a*A*B + b*C```
    - ```A``` is ```m``` by ```k```
    - ```B``` is ```k``` by ```n```
    - ```C``` is ```m``` by ```n```

### SGEMM from cuBLAS

* CUBLAS contains an SGEMM function:
{{cppfrag('09','sgemm/cublas.c','cublas_sgemm')}}

```
m = 320, n = 640, k = 640
Bandwidth: 977.532GB/s
Throughput: 244.383GFlops/s
```
