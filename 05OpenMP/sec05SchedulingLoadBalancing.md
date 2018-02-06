---
title: Scheduling and Load Balancing
---

## Scheduling and Load Balancing


### Number of threads

The number of threads executing an OpenMP 
code is determined by the environmental variable 
`OMP_NUM_THREADS`. 

Normally `OMP_NUM_THREADS` is equal to the number of CPU cores

### Load balancing

Consider our earlier example of a for loop. 10,000,000 iterations split on 4 cores

Two obvious strategies:

* Split in 4 chunks:
    - 2,500,000 iterations for each core
    - Minimal overhead for managing threads
    - Probably a good solution if the cost is independent of thread
    - But what if the cost depends on the thread.
    - One thread might be slower than the rest
* Give each thread one iteration at a time
    - No idling thread
    - But huge overhead

The best solution is probably somewhere in between.


### OpenMP strategies 

OpenMP offers a number of different strategies for 
load balancing set by the following key words. 
The default is static with one chunk per thread. 

* `static:` Iterations are divided into chunks of size `chunk_size` and assigned to threads in round-robin order
* `dynamic`: Each thread executes a chunk of iterations then requests another chunk until none remain
* `guided`: Like dynamic but the chunk size depends on the number of remaining iterations
* `auto`: The decision regarding scheduling is delegated to the compiler and/or runtime system
* `runtime`: The schedule and chunk size are controlled by runtime variables


### Which strategy to use

It is hard to give general advice on the strategy to use. Depends on the problem and platform. Typically needs benchmarking and 
experimentation. 

* If there is little variation in the runtime of iteration static with a large chunk size minimizes overhead
* In other cases it might make sense to reduce the chunk size or use dynamic or guided
* Note that both dynamic and guided comes with additional overhead to schedule the distribution of work
