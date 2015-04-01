---
title: Concepts
---

## Parallel Programming Concepts

### Serial Execution

![Pictures from LLNL Tutorial](session04/figures/serialProblem)


### Parallel Execution

![Pictures from LLNL Tutorial](session04/figures/parallelProblem)


### Some Terminology

* Task = logically discrete set of instructions
* Core = single processing unit, one instruction at a time
* CPU
    * previously synonymous with core
    * now you get multi-core CPU
    * Single unit, fitting single 'socket', single chip (2,4,6,8 core)
* Node = "computer in a box"
* eg. 1000 Nodes, 4 quad core CPU = 16,000 cores.
    
    
### Use Profiling

* WARNING: Before hastily optimising/parallelising
    * Measure sections of your code
    * Obtain evidence on how long each piece takes
    * Consider your deadlines/objectives and how to achieve them
    * Don't over-eagerly parallelise

    
### Amdahl's Law

* Before we start trying to parallelise everything, consider [Amdahl's Law][WikipediaAmdahlsLaw] (Gene Amdahl 1967).

$S(N) = \frac{1}{(1-P) + \frac{P}{N}}$

where N is number of processors, P is proportion [0-1] that can be parallelised.
Note, as N tends to infinity, Speedup tends to 1/(1-P).


### Amdahl's Law - Graph

For example, if 95% can be parallelised, P = 0.95, S = 20.

![Pictures from Wikipedia](session04/figures/AmdahlGraph)


### Amdahl's Law - Example 1

* If P is proportion of code = 0.3, and S = speedup factor = 2 = twice as fast (analagous to 2 processors)

![Pictures from Wikipedia](session04/figures/AmdahlSpeedupUsingS)


### Amdahl's Law - Example 2

* If P1=11%, P2=18%, P3=23%, P4=48%
* P2 sped up by 5
* P3 sped up by 20
* P4 sped up by 1.6

![Pictures from Wikipedia](session04/figures/Amdahl4Components)


### Flynn's Taxonomy

* So, we decide its worth parallelising something.
* First take a view on your hardware
    * Michael Flynn 1966 proposed the following:
        * Single Instruction Single Data (SISD)
        * Single Instruction Multiple Data (SIMD), "data parallel"
        * Multiple Instruction Single Data (MISD)
        * Multiple Instruction Multiple Data (MIMD), "task parallel"


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


### Memory Layout

* In addition to considering processing hardware, need to consider memory
* Various architectures exist, choice determined by coherency/correctness/location

    
### Shared Memory - SMP

* Symmetric Multi-Processing ([SMP][WikipediaSMP]),  eg. x86 multi-core
    * Homogeneous processing cores
    * Address all memory in global address space
    * Uniform Memory Access (UMA) (time)
    * Cache coherency (CC) maintained at hardware level
* Requires code to be multi-threaded

![Pictures from Legion Tutorial](session04/figures/smp2)


### Distributed Memory - DSM

* Truely Distributed Shared Memory ([DSM][WikipediaDSM]) exists
    * Hidden by OS or hardware
    * Allows a single global addressable memory space over networked machines
    * Non-Uniform Memory Access (NUMA)
    * CC-NUMA if cache coherency on
    * See [MOSIX][WikipediaMOSIX] for example
    * Doesn't require code change


### Distributed Memory - Message Passing

* When we say distributed, we normally mean

![Pictures from Legion Tutorial](session04/figures/cluster)


### Distributed Memory

* Advantages
    * Memory scales with processors (i.e. interconnect many SMPs)
    * Rapid local access to memory 
* Disadvantages
    * You must write code to determine how machines are accessed
    * Beware cost of transmitting messages across network
    * Normally no cache coherency across network nodes
    
![Pictures from Legion Tutorial](session04/figures/interconnect)


### Hybrid Distributed-Shared Memory

* Most of the fastest systems are in reality using a hybrid

![Pictures from LLNL Tutorial](session04/figures/hybrid_mem2)

* Increased (easy) scalability is important advantage
* Increased programmer complexity is an important disadvantage


### GPU Accelerator Model

* With GPU processing
    * SIMD on GPU
    * MIMD on multi-core CPU
    * Cost of copying to/from memory
        
![Pictures from Legion Tutorial](session04/figures/gpu)


### Programming Models

* We have now considered
    * Hardware 
    * Memory Architectures
* Programming Models somewhat independent to both!
* People have tried many combinations
* We will now explain most common terms/scenarios

    
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
* Threads exist as part of a process
* Process can contain many threads
* Threads share instructions, execution context and memory
* eg. Web-browser could be a single process (e.g. Firefox)
    * But separate threads download data and refresh the screen
* Implementation:
    * Library: POSIX Threads, pthreads, 1995
    * Compiler: OpenMP for C/C++, 1998
    
    
### Distributed Memory Message Passing

* We saw distributed architectures above
* The programming model is normally defined by Message Passing

![Pictures from LLNL Tutorial](session04/figures/msg_pass_model)


### Hybrid Memory Models

* Or course, there are combinations
    * (Also throw GPU into the mix)
* Its largely down to the developer to write specific code

![Pictures from LLNL Tutorial](session04/figures/hybrid_model)


### SPMD Vs MPMD

* You will also see:
    * Single Program Multiple Data
    * Multiple Program Multiple Data
* High level concepts, built using the above programming models
    * SPMD: Running the same program on many images (e.g. NiftyReg for atlas registration)
    * MPMD: Batch processing a whole workflow of different programs (e.g. LONI Pipeline, FreeSurfer)
    

[LLNLTutorial]: https://computing.llnl.gov/tutorials/parallel_comp/
[WikipediaAmdahlsLaw]: http://en.wikipedia.org/wiki/Amdahl%27s_law
[WikipediaSMP]: http://en.wikipedia.org/wiki/Symmetric_multiprocessing
[WikipediaDSM]: http://en.wikipedia.org/wiki/Distributed_shared_memory
[WikipediaMOSIX]: http://en.wikipedia.org/wiki/MOSIX
[POSIXShared]: http://man7.org/linux/man-pages/man7/shm_overview.7.html
[WikipediaThread]: http://en.wikipedia.org/wiki/Thread_%28computing%29
