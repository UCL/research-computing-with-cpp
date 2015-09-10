---
title: Introduction to Accelerators
---

## Introduction to Accelerators


### What is an accelerator?

* An accelerator is a piece of computing hardware that performs some function faster than is possible in software running on a general purpose CPU
    - an example is the Floating Point Unit inside a CPU that performs calculations on real numbers faster than the Arithmetic and Logic Unit
    - better performance is achieved through concurrency - the ability to perform several operations at once, in parallel
* When an accelerator is separate from the CPU it is referred to as a "hardware accelerator"


### Why would I want to use one?

* Accelerators are designed to execute domain-specific computationally intensive software code faster than a CPU
    - 3D graphics
    - MPEG decoding
    - cryptography


### Using an accelerator within the CPU

* Consider the following code which performs ```y = a*x + y``` on vectors ```x``` and ```y```:

{% idio cpp/saxpy %}

{% fragment saxpy, saxpy.c %}

* Assuming single precision floating point multiply-add is implemented as one CPU instruction the ```for``` loop executes ```n``` instructions (plus integer arithmetic for the loop counter, etc.).


### CPUs as multicore vector processors

* In the mid-1990s, Intel were investigating ways of increasing the multimedia performance of their CPUs without being able to increase the clock speed
* Their solution was to implement a set of registers capable of executing the same operation on multiple elements of an array in a single instruction
    - known as SIMD (Single Instruction, Multiple Data)
    - branded as MMX (64-bit, integer only), SSE (128-bit, integer + floating point) and AVX (256-bit)


### Compiler Autovectorisation

* Modern compilers are able to automatically recognise when SIMD instructions can be applied to certain loops
    - elements are known to be contiguous
    - arrays are not aliased

{% fragment saxpy_fast, saxpy_fast.c %}


### Autovectorisation results

* Running this and timing the invocations produces the following output:

{% code saxpy.out %}

{% endidio %}

* Since Intel's SIMD registers operate on 4 single precision floats at once the first loop executes approximately ```n/4``` instructions
    - resulting in a (near) 4x speedup (Amdahl's Law)
* Can be combined with OpenMP directives (lecture 5) to use SIMD units on multiple cores of the CPU


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
