---
title: CUDA-C
---

## CUDA-C

### CUDA-C Programming Language

* Programming language to write our own GPU kernels
    - based on C99 with extensions
* Compiled with ```nvcc```, Nvidia's compiler for CUDA-C
    - comes with Nvidia CUDA Toolkit
* Contains CPU runtime library with functions to
    - transfer data to/from GPU
    - launch "kernel" functions

### Kernel functions

* Defined with ```__global__``` function attribute
    - run multiple times in parallel on the GPU
    - must return ```void```
* Callable from CPU code

### CUDA Function attributes

* CUDA attributes appear at the start of a function or variable declaration
* ```__device__``` functions can only be called from other ```__device__``` functions or ```__global__``` functions
    - only GPU code is generated
* ```__host__``` functions can only be called from other ```__host__``` functions
    - only CPU code is generated
    - optional modifier
    - can be combined with ```__device__``` to generate code for both CPU and GPU
* ```__global__``` specifies a function that is called from host code but executed on the GPU
    - must return ```void```
    - special calling syntax to specify number of threads, blocks and shared memory

### CUDA Data attributes

* ```__device__``` can also be used to specify a variable that resides in global GPU memory
* ```__const__``` variables are ```const``` and are stored in GPU constant memory
    - they can be accessed directly by the host
* ```__shared__``` variables are stored in shared memory
    - one per block

### Variables for thread indexing

* Within GPU code the following variables are defined and read-only:
    - ```gridDim``` gives the size of the grid
    - ```blockIdx``` gives the index of the current block in the grid
    - ```blockDim``` gives the size of the thread blocks
    - ```threadIdx``` gives the index of the current thread in the block

* These are structs, with e.g. `threadIdx.x` for the x-coordinate.

### Calling CUDA

{{cppfrag('09','cuda/saxpy.cu','CudaCall')}}

The syntax gives first the gridDim, then the blockDim.

When given as integers, 1-D is assumed.

Otherwise, allocate a `dim3`:

``` cuda
dim3 block_dim(32,32,1);
dim3 grid_dim(64,1,1);
kernel<<<grid_dim,block_dim>>>(...);
```

### CUDA SAXPY

* The following code snippet implements ```saxpy``` in CUDA-C:
{{cppfrag('09','cuda/saxpy.cu','saxpy')}}

```
n = 10000, incx = 1, incy = 1
saxpy: 0.010384ms
```


### CUDA SGEMM

* See [github.com/garymacindoe/cuda-cholesky](https://github.com/garymacindoe/cuda-cholesky/blob/master/blas/sgemm.cu) for a CUDA SGEMM optimised for older GPUs

```
m = 320, n = 640, k = 640
Bandwidth: 113.432GB/s
Throughput: 28.358GFlops/s
```

### More on CUDA

* [NVidia CUDA tutorial](http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf)
* [NVidia CUDA Developer Zone](https://developer.nvidia.com/cuda-zone)
