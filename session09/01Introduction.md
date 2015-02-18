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

### An Aside:  Vectorisation

* Consider the following code which performs ```y = a*x + y``` on vectors ```x``` and ```y```:
{{cppfrag('09','saxpy/saxpy.c')}}
* Assuming single precision floating point multiply-add is implemented as one CPU instruction the ```for``` loop executes ```n``` instructions (plus integer arithmetic for the loop counter, etc.).
* Running this and timing the invocations produces the following output:
{{execute('09','saxpy/time-saxpy')}}

### CPUs as vector processors

* In the mid-1990s, Intel were investigating ways of increasing the multimedia performance of their CPUs without being able to increase the clock speed
* Their solution was to implement a set of wide registers capable of executing the same operation on multiple elements of an array in a single instruction
    - also known as vectorisation
* MMX implemented 8 64-bit registers and instructions for operating on pairs of 32-bit integers and single-precision floating point numbers
    - shared the existing CPU registers so required an expensive "context switch" to save and restore register state
* SSE implemented 8 extra 128-bit registers capable of operating on 4 32-bit integers or single-precision floating point numbers or 2 64-bit integers or double-precision floating point numbers
* MMX/SSE only work on contiguous, aligned arrays
    - no gaps between elements
    - address of first element is a multiple of the size of the register

### Compiler Autovectorisation

* Modern compilers are able to automatically recognise when the MMX/SSE instructions can be applied to certain loops
    - currently they need a little help
{{cppfrag('09','saxpy/saxpy-vec.c')}}
* The compiler will vectorise the first loop and insert code to check that ```x``` and ```y``` are correctly aligned before executing
    - else it will execute the unvectorised loop
* Running this and timing the invocations produces the following output:
{{execute('09','saxpy/time-saxpy-vec')}}

### Support for Autovectorisation

* GCC
    - use ```-ftree-loop-vectorize``` to activate
    - use ```-fopt-info-vec``` to check whether loops are vectorised
    - automatically performed with ```-O3```
* ICC
    - use ```-vec``` (Linux/OSX) or ```/Qvec``` (Windows) to activate
    - use ```-vec-report=n```/```/Qvec-report:n``` to check whether loops are vectorised (```n > 0```)
    - automatically performed with ```-O2```/```/O2``` and higher
* Clang
    - use ```-fvectorize``` to activate
    - automatically performed at ```-O2``` and higher
    - no way to see compiler auto-detection
* MSVC
    - activated by default
    - use ```/Qvec-report:n``` to check whether loops are vectorised (```n > 0```)
