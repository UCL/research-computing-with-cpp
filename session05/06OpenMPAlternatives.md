---
title: Alternatives to OpenMP
---

## Alternatives to OpenMP


### Overview

* OpenACC:
    - Similar to OpenMP but intended for accelerators. Lecture 9
* MPI:
    -  For distributed memory systems. Next lecture
* POSIX and Windows threads. Not portable across operation systems
* C++ 11 threads
* [Intel Threading Building Blocks][TBB] C++ only template library
    - Somewhat more complicated. Requires good understanding of templated code
* [Intel Cilk Plus][Cilkplus] C and C++, Intel and GCC >= 4.9 only

### C++11
Simple example: 
{{cppfrag('05','cppthreads/cppthreadsdemo.cc')}}

### Details

* Same problem as first OpenMP example. `std::cout` is not thread safe 
    - Use mutex and locks
* Queues
* Futures and promise
* packed_task

Likely more suited for multi threaded desktop applications than scientific software.

### Cilk Plus

{{cppfrag('05','cilk/fibdemocilk.cc')}}



[TBB]: http://threadingbuildingblocks.org/ 
[Cilkplus]: https://www.cilkplus.org/