---
title: Introduction to OpenMP
---

## OpenMP

### OpenMP basic syntax

* Annotate code with `#pragma omp ...`
    - This instruct the compiler in how to parallize the code
    - `#pragma`s are a instructions to the compiler
    - Not part of the language
    - i.e. `#pragma once` alternative to include guards
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

### CMake Support

CMake knows how to deal with OpenMP, mostly:

```CMake
find_package(OpenMP)

add_program(my_threaded_monster main.cc)
if(OPENMP_FOUND)
  target_compile_options(my_threaded_monster PUBLIC "${OpenMP_CXX_FLAGS}")
  target_link_libraries(my_threaded_monster PUBLIC "${OpenMP_CXX_FLAGS}")
endif()
```


### Hello world

{% code cpp/hello/HelloOpenMP.cc %}

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

{% code cpp/hello/HelloOpenMPSafe.cc %}

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
* `g++ -fopenmp -O3 mycode.cc`


### References

* [OpenMP homepage][OpenMPhomepage]
* [OpenMP cheat sheet][OpenMPcheatsheet]
* [OpenMP specifications][OpenMPSpecs]

[OpenMPhomepage]: http://openmp.org/
[OpenMPcheatsheet]: http://openmp.org/mp-documents/OpenMP-4.0-C.pdf
[OpenMPSpecs]: http://www.openmp.org/mp-documents/OpenMP4.0.0.pdf
[ClangOpenMP]: http://clang-omp.github.io/
