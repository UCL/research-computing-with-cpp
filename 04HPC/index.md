---
title: HPC Concepts
---


## High Performance Computing Overview

### Faster Code 

How can we make a computer go faster?

   * Do instructions faster
   * Do more instructions at once

### Doing Instructions Faster

* How?
    * Reduce wait time. I.E cache misses, data loading, etc.
        * Can be complex and hardware dependant
        * Often automated
    * Write machine code rather than interpreted code
        * JIT does this for you
        * Allows for cool tracing optimisations
    * Increase CPU speed
        * CPU instructions happen faster
        * Data might not **move** faster

### Doing More Instructions at Once

![Dining Philosophers]({% figurepath %}diningphilosophers.png)

[Image variant, original by Benjamin D. Esham / Wikimedia Commons, CC BY-SA 3.0](https://commons.wikimedia.org/w/index.php?curid=56559)

### Concurrency Problems

 * Deadlock
    * All philosophers have one fork each and are waiting for another to appear
 * Starvation
    * A philosopher never gets to eat
 * Race conditions
    * A bug where the order of operations matters.
      This is usually either a algrithm design bug, 
      or two philisophers both using the same fork at the same time.

### Some Better Names

* Philosopher: thread, process, task, etc.
* Fork: resource, lock, etc.
* Spaghetti: Critical section.

### Flynn's Taxonomy

* **SISD**: You doing something on your own
* **SIMD**: You and your clones doing the same things on different inputs
* **MISD**: You and your friends doing different things with the same inputs
* **MIMD**: You and your friends doing unrelated things
* **SIMT**: You and your clones doing the same things on different inputs, but you can choose to skip steps


### Amdahl's Law

You cannot make your code go faster than the non-parallel bit. 


### Background Reading

* This section is based on background reading outside the classroom
* This will save time in the classroom for practical work
* But you do need to know this content

* Read:
    * [Some background history on Wikipedia](http://en.wikipedia.org/wiki/Supercomputer)
    * [Blaise Barney's overview of parallel computing](https://computing.llnl.gov/tutorials/parallel_comp/)

### Essential Reading

* For the exam you will need:
    * [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law)
    * [Flynn's Taxonomy](https://en.wikipedia.org/wiki/Flynn%27s_taxonomy)
* For tutorial's and practical work you will need:
    * Use of unix shell
    * Submitting jobs on a cluster
    * See [RITS HPC Training](http://github-pages.ucl.ac.uk/RCPSTrainingMaterials/)
    * We'll recap this now

### Aim

For the remainder of the course, we need to develop

* Skills to run jobs on a cluster, e.g. Legion.
* A locally installed development environment, so you can develop
* Familiarity with new technologies, OpenMP, MPI, Accelerators, Cloud, so you can make a reasoned choice
