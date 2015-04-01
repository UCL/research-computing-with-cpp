---
title: Background
---

## Overview of high performance computing

### Algorithm development
 
- Consider the case of the Research Programmer:
    + Learns a few languages
    + Spends time developing 'the algorithm'
    + Normally assumes single-threaded
    + Normally assuming von Neumann model of hardware

### Von Neumann architecture

![(a) "John von Neumann, Los Alamos" by LANL. Licensed under Public Domain via Wikimedia Commons. (b) "Von Neumann architecture". Licensed under CC BY-SA 3.0 via Wikimedia Commons.](session04/figures/VonNeumannCombined)

* Proposed by John von Neumann (1903 - 1957) in 1945.
* Based on Turing's work of 1936.
* Instructions and data in same memory.
* Memory bus, causes so called 'Von Neumann bottleneck'.

### Moore's Law

Gordon Moore, co-founder of Intel, 1964: "the number of transistors in a dense integrated circuit doubles approximately every two years"

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

- [Harvard Architecture][http://en.wikipedia.org/wiki/Harvard_architecture]
- [Modified Harvard Architecture][http://en.wikipedia.org/wiki/Harvard_architecture]
- [Multi-level Cache's][http://en.wikipedia.org/wiki/CPU_cache]

But these systems are still limited by heat, power, and cooling.

### Waiting for technology to catch up

Some problems are genuinely too big for existing technology, and can't wait for Moore's Law.

[FreeSurfer][http://freesurfer.net/], an open source software suite for processing and analysing brain MRI images, typically takes 24 hours
- The Alzheimer's Disease Neuroimaging Initiative dataset about 1000 hours.
- Take around 3 years on 1 computer!
- So use batch processing on a cluster.

Gravitational N-body (example from [M.Jones][MJonesTutorial])
- $N$ bodies, takes $N^2$ force calculations
- Best algorithm takes $Nlog_2N$ calculations
- For $10^{12}$ bodies, have $10^{12}ln(10^{12})/ln(2)$ calculations
- So, at $1 \mu sec$, thats $4 x 10^7$ seconds = 1.3 years per step
- So use parallel processing

### World is parallel

Lots of research questions that we might want to simulate are naturally parallel:

![(a) "NGC 4414, a spiral galaxy" by NASA. Licensed under Public Domain via Wikimedia Commons. (b) "Global Forecast System 850 mbar" by the National Weather Service, a branch of National Oceanic and Atmospheric Administration. Licensed under Public Domain via Wikimedia Commons. (c) "Plate tectonics" by the US Geological Survey. Licensed under Public Domain via Wikimedia Commons](session04/figures/galaxy_plate_forecast.png)

### Research computing, parallel computing

- [The Free Lunch Is Over][http://www.gotw.ca/publications/concurrency-ddj.htm]
- So, no more relying on Moore's Law
- Inherent limits in single-core processing
- We must start to learn parallel processing

!["Global Forecast System 850 mbar" by the National Weather Service, a branch of National Oceanic and Atmospheric Administration. Licensed under Public Domain via Wikimedia Commons.](session04/figures/Global_Forecast_850_slice.png)

### History of high performance computing

Recommend reading:

- "[Introduction to high performance computing](http://www.buffalo.edu/content/www/ccr/support/training-resources/tutorials/advanced-topics--e-g--mpi--gpgpu--openmp--etc--/2011-01---introduction-to-hpc--hpc-1-/_jcr_content/par/download/file.res/introHPC-handout-2x2.pdf)" by [Matthew D. Jones][http://www.buffalo.edu/ccr/people/staff/jones.html] at the Center for Computational Research, University at Buffalo, New York.
- The [history on supercomputing on Wikipedia][http://en.wikipedia.org/wiki/History_of_supercomputing]
    
### The first supercomputer
    
The Control Data Corporation 6600 (CDC 6600) is widely considered to be the first 'supercomputer':

- released in 1964
- designed by Seymour Cray
- a factor of 10 quicker than rivals

!["CDC 6600 introduced in 1964" by Steve Jurvetson from Menlo Park, USA. Licensed under CC BY 2.0 via Wikimedia Commons.](session04/figures/CDC6600.png)

### 'Super' is relative

!["Cray-1 Deutsches Museum" by Clemens Pfeiffer. Licensed under CC BY 2.5 via Wikimedia Commons.](session04/figures/Cray1.png)
    
Cray-1, released in 1976 ([M.Jones](http://www.buffalo.edu/content/www/ccr/support/training-resources/tutorials/advanced-topics--e-g--mpi--gpgpu--openmp--etc--/2011-01---introduction-to-hpc--hpc-1-/_jcr_content/par/download/file.res/introHPC-handout-2x2.pdf)): 

- weighed 2400kg
- cost ~$8M
- 160 MFlops

Desktop PC in 2010 ([M.Jones](http://www.buffalo.edu/content/www/ccr/support/training-resources/tutorials/advanced-topics--e-g--mpi--gpgpu--openmp--etc--/2011-01---introduction-to-hpc--hpc-1-/_jcr_content/par/download/file.res/introHPC-handout-2x2.pdf)):

- weighed 5kg
- cost ~$1k
- 48 **G**Flops (quad core, 3Ghz, Intel i7 CPU)
    
### Cray 2

The 1970s to 1980s have been described as the "Cray era". In 1985, around 10 years after Cray 1 was launched, came Cray 2: 

- 1.9 Gflops
- 8 processors

Cray 2 remained the fastest supercomputer until 1990.

!["Cray 2". Licensed under Public Domain via Wikimedia Commons.](session04/figures/Cray2.png)

### The successors

In the 1990s, supercomputers with thousands of processors emerged.

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