---
title: Shared memory parallelism
---

## Shared Memory Parallelism

### OpenMP

* Shared memory only parallelization
* Useful for parallelization on a single cluster node
* MPI next week for inter-node parallelization
* Can write hybrid code with both OpenMP and MPI


### About OpenMP

* Extensions of existing programming languages
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


### Typical use cases

* A loop with independent iterations
* Hopefully a significant part of the execution time
* More complicated if an iterations have dependencies

[OpenMPhomepage]: http://openmp.org/
