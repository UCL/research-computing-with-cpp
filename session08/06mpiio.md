---
title: Parallel IO
---

## Parallel IO

### Serialising on a single process

We know that *any task which is not $O(1/p)$ will eventually dominate the cost as number of processes
increases, preventing scaling.

This is Amdahl's law again.

The master-process-writes approach introduces this problem; IO quickly becomes the dominant part of the
task, preventing weak scaling as problem sizes and processor counts increase.

### Parallel file systems

Supercomputers provide **parallel file systems**. These store each file in multiple "stripes":
one can obtain as much parallelism in IO as there are stripes in files.

To make use of this, it is necessary to use MPI's parallel IO library, MPI-IO.

### Introduction to MPI-IO

MPI-IO works by accessing files as data buffers like core MPI_Send and so on:

{{cppfrag('07','parallel/src/ParallelWriter.cpp','Write')}}

### Opening parallel file

All processes work together to open and create the file:

{{cppfrag('07','parallel/src/ParallelWriter.cpp','Open')}}

### Finding your place

The hard part is synchronising things so that each process writes it's section of the file:

{{cppfrag('07','parallel/src/ParallelWriter.cpp','Seek')}}

### High level research IO libraries

Standard libraries for scientific data formats
such as [HDF5](http://www.hdfgroup.org/HDF5/) support parallel IO. 

You should use these
if you can, as you'll get the benefits of both endianness-portability and parallel IO,
together with built-in metadata support and compatibility with other tools. 

Introduction to
HDF5 or NetCDF is beyond the scope of this course, but familiarising yourself with these
is strongly encouraged!
