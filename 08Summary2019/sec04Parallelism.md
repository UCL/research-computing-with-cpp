---
title: C++ In Parallel
---

## Master Parallelism


### Basic

* Only optimise the most necessary bits
* Use basic timing, profiling or [Amdahls law][Amdahl]
* [Shared Memory Parallelism - OpenMP][shared]
* [Distributed Memory Parallelism - OpenMPI][distributed]


### More advanced - GPU

* Most people do shared memory on single card
* First try a domain specific library that uses GPU e.g [OpenCV][OpenCV]
* Then go more specific if need be: [CUDA][CUDA], [OpenCL][OpenCL], [ArrayFire][ArrayFire]
* Can also be distributed across servers, [OpenMPI][OpenMPI]
* But its hard to deploy, or have access to cards
* You'll need further training courses

    
### More advanced - Cloud

* Cloud is more about availability of nodes/resources
* i.e. same programming, just can scale differently
* If code uses 1 node, batch processing
* If code uses N nodes, very parallel, but difficult to get hold of.
 

[Amdahl]: https://en.wikipedia.org/wiki/Amdahl%27s_law
[shared]: https://www.openmp.org/
[distributed]: https://www.open-mpi.org/
[CUDA]: https://www.nvidia.com/object/cuda_home_temp.html
[OpenCL]: https://www.khronos.org/opencl/
[OpenCV]: https://opencv.org/
[OpenMPI]: https://www.open-mpi.org/