---
title: Introduction to OpenMP
---

## Introduction to OpenMP

### OpenMP

* Shared memory only parallelization
* UMA or NUMA architectures
* Useful for parallelization on a single cluster node
* MPI next week for inter-node parallelization
* Can write hybrid code with both OpenMP and MPI

### About OpenMP

* Extensions of existing programming languages.
* Standardized by international [committee][OpenMPhomepage]
* Support for Fortran, C and C++
* C/C++ uses the same syntax
* Fortran is slightly different

### How it works

* Thread based parallelization
* A master thread starts executing the code.
* Sections of the code is marked as parallel
    - A set of threads are forked and used together with the master thread
    - When the parallel block ends the threads are killed or put to sleep

### Typical use cases

* A loop with independent iterations
* Hopefully a significant part of the execution time
* Remember Amdahl's law
* More complicated if an iterations have dependencies

### OpenMP basic syntax

* Annotate code with `#pragma omp ...`
    - This instruct the compiler in how to parallize the code
    - `#pragma`s are a instructions to the compiler 
    - Not part of the language
    - I.e. `#pragma once` alternative to include guards
    - Compiler will usually ignore pragmas that it doesn't understand
    - All OpenMP pragmas start with `#pragma omp`
* OpenMP must typically be activated when compiling code

### OpenMP library

* OpenMP library:
    - It provides utility functions.
    - `omp_get_num_threads()` ...
    - Use with `#include <omp.h>`

### Compiler support

OpenMP is supported by most compilers, except LLVM/Clang(++)

* OpenMP must typically be activated with a command line flags at compile time. Different for different compilers. Examples:
    - Intel, Linux, Mac  `-openmp`
    - Intel, Windows `/Qopenmp`
    - GCC/G++, `-fopenmp`

A fork of clang with OpenMP [exists][ClangOpenMP]. It might make it into the mainline eventually.

### Hello world

{{cppfrag('05','hello/HelloOpenMP.cc')}}

* `#pragma omp parallel` marks a block is to be run in parallel
* In this case all threads do the same
* No real work sharing

### Issues with this example

* `std::cout` is not thread safe. Output from different threads may be mixed
    - Try running the code
    - Mixed output?
* All threads call `omp_get_num_threds()` with the same result
    - Might be wasteful if this was a slow function
    - Everybody stores a copy of numthreads
    - Waste of memory

### Slightly improved hello world

{{cppfrag('05','hello/HelloOpenMPSafe.cc')}}

### Improvements:

* Use `#pragma omp critical` to only allow one thread to write at a time
    - Comes with a performance penalty since only one thread is running this code at a time
* Use Preprocessor `#ifdef _OPENMP` to only include code if OpenMP is enabled
    - Code works both with and without OpenMP
* Variables defined outside parallel regions
    - Must be careful to tell OpenMP how to handle them
    - `shared`, `private`, `first private`
    - More about this later
* `#pragma omp single`
    - Only one thread calls `get_num_threds()`

### Running OpenMP code for the course

If you have a multicore computer with GCC or other suitable compiler you can run it locally.

Otherwise you can use GCC on aristotle

* `ssh username@aristotle.rc.ucl.ac.uk`
* `module load GCC/4.7.2`
* `g++ -fopenmp -O3 mycode.cc`


### References

* [OpenMP homepage][OpenMPhomepage]
* [OpenMP cheat sheet][OpenMPcheatsheet]
* [OpenMP specifications][OpenMPSpecs]



[OpenMPhomepage]: http://openmp.org/ 
[OpenMPcheatsheet]: http://openmp.org/mp-documents/OpenMP-4.0-C.pdf
[OpenMPSpecs]: http://www.openmp.org/mp-documents/OpenMP4.0.0.pdf
[ClangOpenMP]: http://clang-omp.github.io/