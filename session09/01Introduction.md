---
title: Introduction to Accelerators
---

## Introduction to Accelerators

### What is an accelerator?

* Seperate, dedicated computing device specialised for a particular type of computation:
    - Intel 80387 (Floating Point Math Accelerator for 80386)
    - FPGA (various applications)
    - GPU (accelerating 3D graphics operations)
    - Cryptographic accelerators
* Attached to, and controlled by, a "host" computer.

### Why would I want to use one?

* Can perform certain types of computation faster than a general purpose CPU
* Often with lower power requirements
* Cheaper to buy than a general purpose CPU
    - less applicable and harder to program

### Vectorisation

* Consider the following code which performs ```y = a*x + y``` on vectors ```x``` and ```y```:
{{cppfrag('09','saxpy/saxpy.c','saxpy')}}
* Assuming single precision floating point multiply-add is implemented as one CPU instruction the ```for``` loop executes ```n``` instructions (plus integer arithmetic for the loop counter, etc.).


### CPUs as vector processors

* In the mid-1990s, Intel were investigating ways of increasing the multimedia performance of their CPUs without being able to increase the clock speed
* Their solution was to implement a set of wide registers capable of executing the same operation on multiple elements of an array in a single instruction
    - known as SIMD (Single Instruction, Multiple Data)

### Compiler Autovectorisation

* Modern compilers are able to automatically recognise when SIMD instructions can be applied to certain loops
    - elements are known to be contiguous
    - arrays are not aliased
{{cppfrag('09','saxpy/saxpy_fast.c','saxpy_fast')}}

### Autovectorisation results

* Running this and timing the invocations produces the following output:
{{execute('09','saxpy/saxpy')}}
* Since Intel's SIMD registers operate on 4 single precision floats at once the first loop executes approximately ```n/4``` instructions.


### Support for Autovectorisation

* GCC
    - use ```-ftree-loop-vectorize``` to activate
    - use ```-fopt-info-vec``` to check whether loops are vectorised
    - automatically performed with ```-O3```
* ICC
    - use ```-vec``` (Linux/OSX) or ```/Qvec``` (Windows) to activate
    - use ```-vec-report=n```/```/Qvec-report:n``` to check whether loops are vectorised (```n > 0```)
    - automatically performed with ```-O2```/```/O2``` and higher

### Support for Autovectorisation

* Clang
    - use ```-fvectorize``` to activate
    - automatically performed at ```-O2``` and higher
    - no way to see compiler auto-detection
* MSVC
    - activated by default
    - use ```/Qvec-report:n``` to check whether loops are vectorised (```n > 0```)
