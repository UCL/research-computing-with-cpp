---
title: Parallel Programming
---

## Parallel Programming Concepts

### Serial Execution

![Pictures from LLNL Tutorial](session04/figures/serialProblem)


### Parallel Execution

![Pictures from LLNL Tutorial](session04/figures/parallelProblem)


### Profiling

* This is not a course on profiling ... but ...
* Before hastily optimising
    * Measure sections of your code
    * Obtain evidence on how long each piece takes
    * Consider your deadlines/objectives and how to achieve them
    
    
### Amdahl's Law

* Before we start trying to parallelise everything, consider [Amdahl's Law][WikipediaAmdahlsLaw] (Gene Amdahl 1967).

![Pictures from Wikipedia](session04/figures/AmdahlSpeedup)

where N is number of processors, P is proportion [0-1] that can be parallelised.
Note, as N tends to infinity, Speedup tends to 1/(1-P).


### Amdahl's Law - Graph

For example, if 95% can be parallelised, P = 0.95, S = 20.

![Pictures from Wikipedia](session04/figures/AmdahlGraph)


### Amdahl's Law - Example 1

* If P is proportion of code = 0.3, and S = speedup = 2 = twice as fast.

![Pictures from Wikipedia](session04/figures/AmdahlSpeedupUsingS)


### Amdahl's Law - Example 2

* If P1=11%, P2=18%, P3=23%, P4=48%
* P2 sped up by 5
* P3 sped up by 20
* P4 sped up by 1.6

![Pictures from Wikipedia](session04/figures/Amdahl4Components)


### Flynn's Taxonomy

* So, we decide its worth parallelising something.
* What type of parallisation?
* Michael Flynn 1966 proposed the following:
    * Single Instruction Single Data (SISD)
    * Single Instruction Multiple Data (SIMD), "data parallel"
    * Multiple Instruction Single Data (MISD)
    * Multiple Instruction Multiple Data (MIMD), "task parallel"


### Some Terminology

* Node = "computer in a box"
* core = single processing unit, one instruction at a time
* CPU
    * previously synonymous with core
    * now you get multi-core CPU
    * Single unit, fitting single 'socket'
* Task = logically discrete set of instructions


### SISD

* Single Instruction Single Data
* e.g. Old mainframes, old PC.

![Pictures from LLNL Tutorial](session04/figures/sisd)


### SIMD

* Single Instruction Multiple Data
* e.g. GPU

![Pictures from LLNL Tutorial](session04/figures/simd2)


### MISD

* Multiple Instruction Single Data
* e.g. for fault tolerance

![Pictures from LLNL Tutorial](session04/figures/misd)


### MIMD

* Multiple Instruction Multiple Data
* e.g. distributed computing

![Pictures from LLNL Tutorial](session04/figures/mimd)


### Memory Models

* In addition to considering processing hardware, need to consider memory
* Various models exist, choice determined by coherency/correctness

    
### Shared Memory - SMP

* Address all memory in global address space
* Symmetric Multi-Processing ([SMP][WikipediaSMP])
    * Homogeneous processing cores
    * Uniform Memory Access (UMA) (time)
    * Cache coherency (CC) maintained at hardware level
* Requires code to be multi-threaded
* Examples: x86 multi-core

![Pictures from Legion Tutorial](session04/figures/smp2)


### Distributed Memory - DSM

* Distributed Shared Memory ([DSM][WikipediaDSM]) exists
    * Hidden by OS or hardware
    * Allows a single addressable memory space
    * Non-Uniform Memory Access (NUMA)
    * CC-NUMA if cache coherency on
    * See [MOSIX][WikipediaMOSIX] for example


### Distributed Memory - Message Passing

* When we say distributed, we normally mean

![Pictures from Legion Tutorial](session04/figures/cluster)


### Distributed Memory - Cost

* Advantages
    * Memory scales with processors (i.e. interconnect SMPs)
    * Rapid local access to memory without maintaining global CC
* Disadvantages
    * Programmers must write code to determine how machines are accessed
    * Beware cost of transmitting messages across network

![Pictures from Legion Tutorial](session04/figures/interconnect)


### Hybrid Distributed-Shared Memory

* Most of the fastest systems are in reality using a hybrid

![Pictures from Legion Tutorial](session04/figures/hybrid_mem2)

* Increased scalability is important advantage
* Increased programmer complexity is an important disadvantage


### GPU Accelerator Model

* For completeness
* With GPU processing
    * Cost of copying to/from memory
    * SIMD on GPU
    * MIMD on multi-core CPU
    
![Pictures from Legion Tutorial](session04/figures/gpu)


### Programming Models

* We have now considered
    * Hardware 
    * Memory Architectures
* Programming Models somewhat independent to both!
* People have tried many combinations
* We will now explain common terms/scenarios
    
### Shared Memory No Threads

* Separate Processes, Shared Memory
* Stand-alone
    * [POSIX standard for shared memory][POSIXShared]
* Advantages:
    * No message passing
* Disadvantages: 
    * No concept of ownership


### Shared Memory With Threads

* [Thread][WikipediaThread]: "of execution is the smallest sequence of programmed instructions that can be managed independently by a scheduler"
* Process can contain many threads
* Threads exist as part of a process
* Threads share instructions and execution context
* eg. Web-browser could be a single process (e.g. Firefox)
    * But separate threads download data and refresh the screen
* POSIX Threads, pthreads, 1995
* OpenMP, C/C++, 1998
    
    
### Distributed Memory Message Passing

* We saw distributed architectures above
* The programming model is normally defined by Message Passing

![Pictures from LLNL Tutorial](session04/figures/msg_pass_model)


### Hybrid Memory Models

* Or course, there are combinations
* Its largely down to the developer to write specific code
* Also throw GPU into the mix

![Pictures from LLNL Tutorial](session04/figures/hybrid_model)


### SPMD Vs MPMD

* You will also see:
    * Single Program Multiple Data
    * Multiple Program Multiple Data
* High level concepts, built using the above programming models
    * SPMD: Running the same program on many images (e.g. FreeSurfer)
    * MPMD: Batch processing a whole workflow of different programs (e.g. LONI Pipeline)
    

### Understand Your Task


![Pictures from LLNL Tutorial](session04/figures/hybrid_model)

* Profile (see above)
* Are (sub-)tasks parallelisable?
* Identify hotspots and bottlenecks?
* Use optimised libraries


### Input Output

![Pictures from LLNL Tutorial](session04/figures/memoryAccessTimes)

* Reduce I/O as far as possible
* Use parallel FS if possible
* Use SSD discs if possible


### Partitioning

* Partition by:
    * data
        * ITK, chop image into 8ths, 8 threads
        * GPU same operation on each pixel
    * task
        * Sub-divide tasks
        * Which run in parallel
        * How to re-join, synchronise?
        

### Load Balancing / Granularity

* Granularity = size of each task
* Consider how to keep ALL CPU's busy ALL the time
* Depends
    * Hardware model
    * memory model
    * programming model 
    * communication model
    

### Other Limits/Costs

* Specificity - develop a specific customised version for a specific task
* Complexity - design, coding, testing, understandability
* Portability - see standardisation, OpenMP, MPI, Posix Threads.
* Resource Requirements - CPU/Memory/Network/Disk
* Scalability - related to Amdahl's law, memory overhead, communications overhead, synchronisation cost.

### Choice Of Technology

* Need to consider all above factors
* Rest of course
    * OpenMP - SIMD, shared memory, multi-threading, compiler directive
    * MPI - MIMD, distributed memory, multi-process, programmatic
    * GPU - SIMD, GPU memory, multi-threading, custom kernel
* Gives reasonable overview


[LLNLTutorial]: https://computing.llnl.gov/tutorials/parallel_comp/
[WikipediaAmdahlsLaw]: http://en.wikipedia.org/wiki/Amdahl%27s_law
[WikipediaSMP]: http://en.wikipedia.org/wiki/Symmetric_multiprocessing
[WikipediaDSM]: http://en.wikipedia.org/wiki/Distributed_shared_memory
[WikipediaMOSIX]: http://en.wikipedia.org/wiki/MOSIX
[POSIXShared]: http://man7.org/linux/man-pages/man7/shm_overview.7.html
[WikipediaThread]: http://en.wikipedia.org/wiki/Thread_%28computing%29
