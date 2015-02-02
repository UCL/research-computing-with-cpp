---
title: Background
---

## Overview of HPC

### Von Neumann Architecture

![Pictures from wikipedia](session04/figures/VonNeumannCombined)

* John von Neumann (1903 - 1957) proposed in 1945.
* Based on Turing's work in 1936.
* Instructions and Data in same memory.
* Memory bus, causes so called Von Neuman bottleneck.


### Algorithm Development
 
* For interested readers
    * [Harvard Architecture][WikipediaHarvardArch]
    * [Modified Harvard Architecture][WikipediaModifiedHarvardArch]
    * [Multi-level Cache's][WikipediaCache]
* Consider the case of the Research Programmer:
    * Learns a few languages
    * Spends time developing 'the algorithm'
    * Normally assumes single-threaded
    * Normally assuming von Neumann model of hardware


### Moore's Law

* Gordon Moore, co-founder Intel, 1964:
    * "the number of transistors in a dense integrated circuit doubles approximately every two years"

![Pictures from wikipedia](session04/figures/TransistorCount)


### Research Can't Wait

* Some problems are genuinely too big
* Can't wait for Moore's Law to work 

![Picture from M.Jones](session04/figures/GravitationalProblem)

   
### Physical Limitations

* [Herb Sutter's, The Free Lunch Is Over][HerbFreeLunch]

![Picture from Sutter](session04/figures/CPUPerf)

* Due to clockspeed, power requirements, cooling, not getting much more processing power per CPU.
* Wire delays, memory access times hard to improve.
* Manufacturers are turning to multi-core.


### World Is Parallel

* Lots of things that we might want simulate are naturally parallel

![Picture from M.Jones](session04/figures/realWorldCollage1)
![Picture from M.Jones](session04/figures/realWorldCollage2)


### Research Computing, Parallel Computing

* [The Free Lunch Is Over][HerbFreeLunch]
* So, no more relying on Moore's Law
* Inherent limits in single-core processing
* We must start to learn parallel processing

![Picture from M.Jones](session04/figures/noaaforcast)

### History of HPC

* Recommend reading
    * Notes by [M.Jones][MJonesTutorial]
    * [History page on Wikipedia][WikipediaHistory]
    * A few highlights to inspire you
    
    
### History - 1
    
* 1964, Seymour Cray, Control Data Corporation (CDC) 6600
* Factor 10 quicker than rivals, so considered first 'super computer'

![Picture from wikipedia](session04/figures/CDC6000)


### Its Relative

* From [Wikipedia][WikiPediaSuperComputer], ![Picture from wikipedia](session04/figures/440px-Cray-1-deutsches-museum)
    
* Cray-1, 1976, 2400kg, $8M, 160MFlops ([M.Jones][MJonesTutorial]).
* Desktop PC, 2010, 5kg, $1k, 48GFlops ([M.Jones][MJonesTutorial]).
    * (quad core, 3Ghz, Intel i7 CPU)

    
### History - 2

* Cray 1, 1976 (see above)
* Cray 2, 1985, 1.9Gflops, 8 processors, fastest until 1990.

![Picture from wikipedia](session04/figures/Cray2)


### History - 3

* 1990's - clusters
* 1993 Beowulf Cluster - commodity PC's
* 1997, ASCI Red, first TFlop/s, 9000 nodes of Pentium Pro
* 2004 - 2007, IBM Blue Gene, 70-472 TFlop/s, 60,000 processors

![Picture from wikipedia](session04/figures/480px-IBM_Blue_Gene_P_supercomputer)


### Top500.org

* See top500.org for list
* Number 1 = TIANHE-2, 3.1M Cores, 1,024,000 GB memory!, 33PFlop/s!
* You can get access to:
    * Legion: 7500 CPU cores + 7168 CUDA cores, 32 cores per job
    * Iridis: 12,000 cores, 900 nodes, 100 nodes (1200 cores) per job
    * Emerald: 372 NVIDIA Tesla, 114TFlop/s
    * Archer: 25th, 1.6PFlop/s, 
    
[MJonesTutorial]: http://www.buffalo.edu/content/www/ccr/support/training-resources/tutorials/advanced-topics--e-g--mpi--gpgpu--openmp--etc--/2011-01---introduction-to-hpc--hpc-1-/_jcr_content/par/download/file.res/introHPC-handout-2x2.pdf
[WikipediaHarvardArch]: http://en.wikipedia.org/wiki/Harvard_architecture
[WikipediaModifiedHarvardArch]: http://en.wikipedia.org/wiki/Modified_Harvard_architecture
[WikipediaCache]: http://en.wikipedia.org/wiki/CPU_cache
[HerbFreeLunch]: http://www.gotw.ca/publications/concurrency-ddj.htm
[WikipediaHistory]: http://en.wikipedia.org/wiki/History_of_supercomputing




