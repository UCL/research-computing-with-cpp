---
title: CUDA Thrust
---

## CUDA Thrust

### CUDA Thrust

* CUDA Thrust is a C++ template library of parallel algorithms and data structures based on the STL
    - sort, scan, transform and reduction algorithms
    - uses ```host_vector``` and ```device_vector``` templated types modelled on ```std::vector```
    - supports GPUs via CUDA and CPUs via OpenMP through the same interface
    - all headers in the ```thrust``` subdirectory
    - all data structures and algorithms declared in the ```thrust``` namespace
* Open source using the Apache License
    - supplied with the Nvidia CUDA Toolkit
    - source code on GitHub

### Vector types

* ```template <class T> class thrust::host_vector```
    - declared in ```#include <thrust/host_vector.h>```
    - similar to ```std::vector``` using CPU memory
* ```template <class T> class thrust::device_vector```
    - declared in ```#include <thrust/device_vector.h>```
    - similar to ```std::vector``` but using GPU memory

### Vector types: Constructors

* The following code fragment shows the three constructors available for ```host_vector```

{{cppfrag('09','thrust/vector_constructors.cu')}}

* ```device_vector```s have the same constructors but with storage allocated in GPU memory

### Vector types: Copy Constructors

* There are four copy constructors for both ```device_vector``` and ```host_vector```
    - copy from a vector of the same template type
    {{cppfrag('09','thrust/vector_copy.cu','copy')}}

    - copy from a vector of a different template type
    {{cppfrag('09','thrust/vector_copy.cu','template_copy')}}

    - copy a device_vector from a host_vector (or vice versa)
    {{cppfrag('09','thrust/vector_copy.cu','transfer_copy')}}

    - create a copy of an STL vector
    {{cppfrag('09','thrust/vector_copy.cu','stl_copy')}}

* Each copy constructor also has a corresponding assignment operator

### Vector types: Accessors

* The ```[]``` operator has been overloaded for ```device_vector``` and ```host_vector```
{{cppfrag('09','thrust/vector_assignment.cu')}}
* Be careful when accessing the elements of a ```device_vector``` from host code as each one performs a transfer from GPU memory

### Algorithms

* In addition to the two vector data types, Thrust also implements several templated algorithms
    - transform
    - reduce
    - transform_reduce
    - sort
    - search

### Algorithms: transform

* The transform algorithm is declared in ```thrust/transform.h```
{{cppfrag('09','thrust/transform.cu')}}
* ```transform``` applies a unary_op functor to the elements between ```first``` and ```last``` and stores them in ```result```

### Algorithms: reduce

* The reduction algorithm is declared in ```thrust/reduce.h```
{{cppfrag('09','thrust/reduce.cu')}}
* ```reduce``` applies a ```binary_op``` functor to the elements between ```first``` and ```last``` starting with ```init``` and returns the result

### Algorithms: transform/reduce

* The transform/reduce algorithm is declared in ```thrust/transform_reduce.h```
    - it combines the transform and reduce algorithms
    - first elements are transformed using ```unary_op```
    - then reduced using ```binary_op```
{{cppfrag('09','thrust/transform_reduce.cu')}}

### Thrust SAXPY

* We can use the ```transform``` algorithm from Thrust to implement our SAXPY kernel
{{cppfrag('09','saxpy/thrust.cu','saxpy')}}

```
Bandwidth: 15.7344GB/s
Throughput: 7.86722GFlops/s
```

### Thrust SGEMM

* Likewise we can use the ```transform_reduce``` algorithm to implement an SGEMM kernel
{{cppfrag('09','sgemm/thrust.cu','sgemm')}}

```
Bandwidth: 18.0532GB/s
Throughput: 9.02659GFlops/s
```

### Further Information

* GitHub repository at https://github.com/thrust/thrust
* Documentation at https://thrust.github.io
* Tuturial at http://docs.nvidia.com/cuda/thrust/
