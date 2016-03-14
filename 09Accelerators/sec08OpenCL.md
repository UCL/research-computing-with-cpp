---
title: OpenCL
---

## OpenCL

### Why OpenCL?

CUDA only works on NVidia cards.

OpenCL is an open standard which works on all graphics cards,
and also on [FPGAs](https://en.wikipedia.org/wiki/Field-programmable_gate_array).

It uses similar models - copying memory from device to host,
launch of kernels with specified thread count and block size. Block size is called "work group size" in OpenCL. (Roughly)

### Why not OpenCL

However, there is no equivalent of the Thrust C++ library,
and OpenCL kernels may only include C code.
(This should change with OpenCL 2.0, but hardware support for
this is poor.)

Further, compilation of OpenCL code requires
the use of a cumbersome C interface to dispatch source code for compilation and linkage.

### OpenCL example

See [Legion Scaffold](https://github.com/UCL-RITS/Legion-Fabric-Scaffold/blob/opencl/src/cl_main.cpp).
