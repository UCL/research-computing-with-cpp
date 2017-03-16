---
title: MPI in practice
---

## MPI in practice




### Specification and implementation

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

### Programming and running

* an MPI program is executed with ``mpiexec -n N [options] nameOfProgram [args]``

* MPI programs call methods from the mpi library

``` cpp
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
               MPI_Comm comm)
```


* vendors provide wrappers (mpiCC, mpic++) around compilers.
  Wrappers point to header file location and link to right libraries.
  MPI program can be (easily) compiled by substituting ``(g++|icc) -> mpiCC``

### Hello, world!: hello.cc

{% code cpp/hello.cc %}

### Hello, world!: CMakeLists.txt

``` CMake
find_package(MPI REQUIRED)

add_executable(hello hello.cc)
target_include_directories(hello SYSTEM PUBLIC ${MPI_INCLUDE_DIRS})
target_link_libraries(hello PUBLIC ${MPI_LIBRARIES})
```

### Hello, world!: compiling and running

On aristotle.rc.ucl.ac.uk:

- load modules:
  ``module load GCC/4.7.2 OpenMPI/1.6.4-GCC-4.7.2``
  ``module load cmake/2.8.10.2``
- create files "hello.cc" and "CMakeLists.txt" in some directory
- create build directory ``mkdir build && cd build``
- run cmake and make ``cmake .. && make``
- run the code ``mpiexec -n 4 hello``

### Hello, world! dissected

- MPI calls *must* appear beween ``MPI_Init`` and ``MPI_Finalize``
- Groups of processes are handled by a communicator. `MPI_COMM_WORLD` handles
    the group of all processes.
    ![]({% figurepath %}world.png)

- Size of group and rank (order) of process in group
- By *convention*, process of rank 0 is *special* and called *root*

### MPI with CATCH

Running MPI unit-tests requires ``MPI_Init`` and ``MPI_Finalize`` before and after the
test framework (*not* inside the tests).

{% code cpp/helloCatch.cc %}
