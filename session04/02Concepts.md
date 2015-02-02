---
title: Parallel Programming
---

## Parallel Programming Concepts

### Serial Execution

![Pictures from LLNL Tutorial](session04/figures/serialProblem)


### Parallel Execution

![Pictures from LLNL Tutorial](session04/figures/parallelProblem)


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

* Programmers must write code to determine how machines are accessed
* Beware cost of transmitting messages across network

![Pictures from Legion Tutorial](session04/figures/interconnect)


### Hybrid Distributed-Shared Memory

* Most of the fastest systems are in reality using a hybrid

![Pictures from Legion Tutorial](session04/figures/hybrid_mem2)


### Programming Models



### Shared Memory No Threads

### Shared Memory With Threads

### Distributed Memory Message Passing

### Data Parallel

### SPMD

### MPMD

### Caching

### Cache Coherency

### Understand The Problem

### Patitioning

### Synchronisation

### Data

### Load Balancing

### Granularity

### Input Output

### Other Limits/Costs

* Complexity - design, coding, testing, understandability
* Portability - see standardisation, OpenMP, MPI, Posix Threads.
* Resource Requirements - CPU/Disk/Network
* Scalability - related to Amdahl's law, memory overhead, communications overhead, synchronisation cost.

### Choice Of Technology


[LLNLTutorial]: https://computing.llnl.gov/tutorials/parallel_comp/
[WikipediaAmdahlsLaw]: http://en.wikipedia.org/wiki/Amdahl%27s_law
[WikipediaSMP]: http://en.wikipedia.org/wiki/Symmetric_multiprocessing
[WikipediaDSM]: http://en.wikipedia.org/wiki/Distributed_shared_memory
[WikipediaMOSIX]: http://en.wikipedia.org/wiki/MOSIX