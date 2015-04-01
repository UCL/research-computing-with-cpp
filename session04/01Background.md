---
title: Background
---

## Overview of high performance computing

### Algorithm development
 
* Consider the case of the Research Programmer:
    * Learns a few languages
    * Spends time developing 'the algorithm'
    * Normally assumes single-threaded
    * Normally assuming von Neumann model of hardware

### Von Neumann architecture

![(a) "John von Neumann, Los Alamos" by LANL. Licensed under Public Domain via Wikimedia Commons. (b) "Von Neumann architecture". Licensed under CC BY-SA 3.0 via Wikimedia Commons.](session04/figures/VonNeumannCombined)

* Proposed by John von Neumann (1903 - 1957) in 1945.
* Based on Turing's work of 1936.
* Instructions and data in same memory.
* Memory bus, causes so called 'Von Neumann bottleneck'.

### Moore's Law

* Gordon Moore, co-founder of Intel, 1964:
    * "the number of transistors in a dense integrated circuit doubles approximately every two years"

!["Transistor Count and Moore's Law 2011" by Wgsimon. Licensed under CC BY-SA 3.0 via Wikimedia Commons.](session04/figures/TransistorCount.png)


### Physical limitations

![CPU scaling showing transistor density, power consumption, and efficiency. Chart from The Free Lunch Is Over: A Fundamental Turn Toward Concurrency in Software. Copyright Sutter, 2009.](session04/figures/CPUPerf)

Manufacturers are [turning to multi-core systems](http://www.gotw.ca/publications/concurrency-ddj.htm), for reasons including:
- clockspeed
- power requirements
- cooling
- wire delays
- memory access times
 

### More caches?

It might be tempting to think that caches solve everything. For interested readers:

- [Harvard Architecture][WikipediaHarvardArch]
- [Modified Harvard Architecture][WikipediaModifiedHarvardArch]
- [Multi-level Cache's][WikipediaCache]

But these systems are still limited by heat, power, and cooling.

### Waiting for technology to catch up

Some problems are genuinely too big for existing technology, and can't wait for Moore's Law.

- [FreeSurfer][FreeSurfer] typically takes 24 hours
    + ADNI dataset about 1000
    + Take approx 3 years on 1 computer!
    + So use batch processing on a cluster

- Gravitational N-body (example from [M.Jones][MJonesTutorial])
    + $N$ bodies, takes $N^2$ force calculations
    + Best algorithm takes $Nlog_2N$ calculations
    + For $10^{12}$ bodies, have $10^{12}ln(10^{12})/ln(2)$ calculations
    + So, at $1 \mu sec$, thats $4 x 10^7$ seconds = 1.3 years per step
    + So use parallel processing

### World is parallel

Lots of research questions that we might want to simulate are naturally parallel:

![Pictures from wikipedia](session04/figures/ParallelComputingExamples)

### Research computing, parallel computing

- [The Free Lunch Is Over][HerbFreeLunch]
- So, no more relying on Moore's Law
- Inherent limits in single-core processing
- We must start to learn parallel processing

![Picture from Legion Tutorial](session04/figures/noaaforcast)

### History of high performance computing

Recommend reading:
- "[Introduction to high performance computing](http://www.buffalo.edu/content/www/ccr/support/training-resources/tutorials/advanced-topics--e-g--mpi--gpgpu--openmp--etc--/2011-01---introduction-to-hpc--hpc-1-/_jcr_content/par/download/file.res/introHPC-handout-2x2.pdf)" by [Matthew D. Jones][http://www.buffalo.edu/ccr/people/staff/jones.html] at the Center for Computational Research, University at Buffalo, New York.
- The [history on supercomputing on Wikipedia][http://en.wikipedia.org/wiki/History_of_supercomputing]
    
### The first supercomputer
    
- 1964, Seymour Cray, Control Data Corporation (CDC) 6600
- Factor 10 quicker than rivals, so considered first 'super computer'

!["CDC 6600 introduced in 1964" by Steve Jurvetson from Menlo Park, USA. Licensed under CC BY 2.0 via Wikimedia Commons.](session04/figures/CDC6600.png)

### Speed is relative

!["Cray-1 Deutsches Museum" by Clemens Pfeiffer. Licensed under CC BY 2.5 via Wikimedia Commons.](session04/figures/Cray1.png)
    
* Cray-1, 1976, 2400kg, $8M, 160MFlops ([M.Jones][MJonesTutorial]).
* Desktop PC, 2010, 5kg, $1k, 48GFlops ([M.Jones][MJonesTutorial]).
    * (quad core, 3Ghz, Intel i7 CPU)
    
### Cray 2

In 1985, around 10 years after Cray 1 was launched, came Cray 2: 

- 1.9 Gflops
- 8 processors

Cray 2 remained the fastest supercomputer until 1990.

!["Cray 2". Licensed under Public Domain via Wikimedia Commons.](session04/figures/Cray2.png)

### The successors

* 1990s - clusters
* 1993 Beowulf Cluster - commodity PCs
* 1997, ASCI Red, first TFlop/s, 9000 nodes of Pentium Pro
* 2004 - 2007, IBM Blue Gene, 70-472 TFlop/s, 60,000 processors

!["IBM Blue Gene P Supercomputer" by Argonne National Laboratory. Licensed under CC BY-SA 2.0 via Wikimedia Commons.](session04/figures/IBM_Blue_Gene.png)

### The supercomputer league table

[TOP500](http://top500.org) maintains a list of the fastest supercomputers. In November 2014, the fastest supercomputer was Tianhe-2 (MilkyWay-2), a supercomputer developed by China's National University of Defense Technology.

Tianhe-2 runs Linux and is reported to have:

- 3,120,000 cores
- 1,024,000 GB of memory!
- a performance of 33.86 petaflop/s (quadrillions of calculations per second)

At UCL, you can get access to systems including:

- Legion: 7500 CPU cores + 7168 CUDA cores, 32 cores per job
- Iridis: 12,000 cores, 900 nodes, 100 nodes (1200 cores) per job
- Emerald: 372 NVIDIA Tesla, 114TFlop/s
- Archer: 25th, 1.6PFlop/s (requires a grant)
    
[MJonesTutorial]: http://www.buffalo.edu/content/www/ccr/support/training-resources/tutorials/advanced-topics--e-g--mpi--gpgpu--openmp--etc--/2011-01---introduction-to-hpc--hpc-1-/_jcr_content/par/download/file.res/introHPC-handout-2x2.pdf
[WikipediaHarvardArch]: http://en.wikipedia.org/wiki/Harvard_architecture
[WikipediaModifiedHarvardArch]: http://en.wikipedia.org/wiki/Modified_Harvard_architecture
[WikipediaCache]: http://en.wikipedia.org/wiki/CPU_cache
[HerbFreeLunch]: http://www.gotw.ca/publications/concurrency-ddj.htm
[WikipediaHistory]: http://en.wikipedia.org/wiki/History_of_supercomputing
[FreeSurfer]: http://freesurfer.net/