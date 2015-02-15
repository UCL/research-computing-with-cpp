---
title: MPI in practice
---

## MPI in practice

### Paradigm

* many processes, each with their own data

    ![](session06/figures/many.png)

* each process is independent
* processes can send messages to one another


### Specification and Implementation

* in practice, we use MPI, the [Message Passing Interface](http://en.wikipedia.org/wiki/Message_Passing_Interface)
* MPI is a *specification* for a *library*
* It is implemented by separate vendors/open-source projects
     - [OpenMPI](http://www.open-mpi.org/)
     - [mpich](http://www.mpich.org/)
* It is a C library with many many bindings:
     - Fortran (part of official MPI specification)
     - Python: [boost](http://www.boost.org/doc/libs/1_55_0/doc/html/mpi/python.html), [mpi4py](http://mpi4py.scipy.org/)
     - R: [Rmpi](http://cran.r-project.org/web/packages/Rmpi/index.html)
     - c++: [boost](http://www.boost.org/doc/libs/1_57_0/doc/html/mpi.html)

### Programming and Running

* an MPI program is executed with ``mpiexec -n N [options] nameOfProgram [args]``

* MPI programs call methods from the mpi library

~~~{.c++}
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
               MPI_Comm comm)
~~~


* vendors provide wrappers (mpiCC, mpic++) around compilers.
  Wrappers point to header file location and link to right libraries.
  MPI program can be (easily) compiled by substituting ``(g++|icc) -> mpiCC``

* in cmake

~~~{.cmake}
find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
target_library(some_mpi_target ${MPI_C_LIBRARIES})
~~~

