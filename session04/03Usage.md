---
title: Usage
---

## Usage

### Understand your task

![Pictures from LLNL Tutorial](session04/figures/memoryAccessTimes)

* Profile, gather timings on each task/code section
* Reduce I/O as far as possible
* Use parallel FS if possible
* Use SSD discs if possible


### Identify candidates

* Identify hotspots and bottlenecks?
* Use optimised libraries
* Are (sub-)tasks parallelisable?


### Partitioning

* Partition tasks by:
    * data
        * ITK, chop image into 8ths, so 8 threads
        * GPU same operation on each pixel
    * task
        * Sub-divide tasks
        * Which tasks can run in parallel
        * How to re-join, synchronise?
        
        
### Load balancing/granularity

![Pictures from LLNL Tutorial](session04/figures/hybrid_model)

* Granularity = size of each task
* Consider how to keep ALL cores busy ALL the time
    

### Other limits/costs

* If you parallelise, be aware:
    * Specificity - you develop a specific customised version for a specific task
    * Complexity - cost of design, coding, testing, understandability
    * Portability - use standardisation, OpenMP, MPI, Posix Threads.
    * Resource Requirements - balance CPU/Memory/Network/Disk
    * Scalability - related to Amdahl's law, memory overhead, communications overhead, synchronisation cost.


### Choice of technology

* Depends
    * Hardware architecture
    * Memory architecture
    * Programming model 
    * Communication model
* Rest of course
    * OpenMP - SIMD, shared memory, multi-threading, compiler directive
    * MPI - MIMD, distributed memory, multi-process, programmatic
    * GPU - SIMD, GPU memory, multi-threading, custom kernel
* So we cover a reasonable overview
