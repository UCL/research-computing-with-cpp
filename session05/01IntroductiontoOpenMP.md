---
title: Introduction to OpenMP
---

## Introduction to OpenMP

### OpenMP

* Annotate code with `#pragma omp ...`
    - This instruct the compiler in how to parallize the code
    - `#pragma` is a general way of providing instructions to the compiler
    - I.e. `#pragma once` etc.
    - All OpenMP Pragmas start with `#pragma omp`
* In addition there is omp library.
    - It provides utility functions 
    - Use with `#include <omp.h>`

### Hello World

Simple example.
{{cppfrag('05','hello/HelloOpenMP.cc')}}

* `#pragma omp parallel` marks a block is to be run in parallel.
* In this case all threads do the same.
* `std::cout` is not thread safe. Output from different threads may be mixed
    - Try running the code.
    - Mixed output? 
* All threads call omp_get_num_threds() with the same result.
    - Might be wasteful if this was a slow function.
    - Everybody stores a copy of numthreads
    - Waste of memory

### Slightly improved Hello World

{{cppfrag('05','hello/HelloOpenMPSafe.cc')}}

Improvements:

* Use `#pragma omp critical` to only allow one thread to write at a time.
    - Comes with a performance penalty since only one thread is running this code at a time.
* Use Preprocessor `#ifdef _OPENMP` to only compile code if OpenMP is enabled. 
    - Code works both with and without OpenMP 
* Variables defined outside parallel regions.
    - Must be careful to tell OpenMP how to handle them.
    - `shared`, `private`, `first private` etc.
* `#pragma omp single`
    - Only one thread calls `get_num_threds()`


### References

[OpenMP homepage][OpenMPhomepage]
[OpenMP cheat sheet][OpenMPcheatsheet]





[OpenMPhomepage]: http://openmp.org/ 
[OpenMPcheatsheet]: http://openmp.org/mp-documents/OpenMP-4.0-C.pdf