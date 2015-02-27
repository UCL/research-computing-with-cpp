---
title: Using a GPU as an Accelerator
---

## Using a GPU as an Accelerator

### GPUs for 3D graphics

* 3D graphics rendering involves lots of operations on 3 and 4 dimensional vectors
    - positions of vertices (x, y, z)
    - colour and transparency of pixels (RGBA)
* Realtime graphics rendering needs to be fast so accuracy is often sacrificed
    - GPUs are good at 32-bit integer and single precision floating point arithmetic
    - not as good at 64-bit integer and double precision floating point
* 3D graphics operations are independent of one another
    - and can be performed in parallel

### GPUs as multicore vector processors

* In order to meet the performance needs of high resolution realtime 3D graphics GPUs were developed to be high-throughput parallel processors
* GPUs are  multicore vector processors similar to CPUs but with much faster memory access
+-----------------+---------+---------+
|                 |   CPU   |   GPU   |
+=================+=========+=========+
|No of cores      |   <10   |   1000s |
+-----------------+---------+---------+
|SIMD width       | 256 bits|1024 bits|
+-----------------+---------+---------+
|Memory bandwidth | ~25 GB/s|~500 GB/s|
+-----------------+---------+---------+

### General Purpose Programming for GPUs

* With the introduction of programmable vertex shaders for 3D graphics it became possible to "trick" GPUs into performing other types of computation
    - such as matrix multiply
* This involved uploading data as an image into GPU memory, running a custom vertex program and reading the results back
* Several frameworks were developed to simplify this
    - BrookGPU
    - Accelerator
* Nvidia CUDA provides a software toolkit that allows programmers to use GPUs as parallel computing devices rather than graphics processors
    - compiler for GPU code written in CUDA-C
    - runtime library to load code and data onto GPU and run it

### Matrix Multiply on a GPU

* Given the differences in numbers of cores and SIMD width a GPU should outperform a CPU