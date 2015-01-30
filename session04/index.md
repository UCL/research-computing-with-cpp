---
title: HPC Concepts
---

## Why HPC?

### What is HPC?

* HPC: High Performance Computing
*   refers to "any computational activity requiring more than a single computer to execute a task."
* From [this tutorial][MJonesTutorial]
    "HPC requires substantially (more than an order of magnitude) more computational resources 
    than are available on current workstations, and typically require concurrent (parallel) computation."
* So,
    * big + parallel

  
### Its Relative

* From [Wikipedia][WikiPediaSuperComputer], ![Cray-1 1976](session04/figures/440px-Cray-1-deutsches-museum)
    
* Cray-1, 1976, 2400kg, $8M, 160MFlops ([M.Jones][MJonesTutorial]).
* Desktop PC, 2010, 5kg, $1k, 48GFlops ([M.Jones][MJonesTutorial]).
    * (quad core, 3Ghz, Intel i7 CPU)

    
### Why HPC in Research?

* Its the nature of research
    * Always tackle bigger problems
    * Driving the limits of hardware and software

    
### Why Teach HPC?

* Read [Herb Sutter's Article][SutterWTTJ]
* All major devices have multiple cores
    * Phone ([Apple A8][AppleA8] dual core), 
    * Tablet ([Apple A8][AppleA8] dual core),
    * Laptop (Apple MBP, 2.5Ghz, quad core),
    * Desktop (Apple Mac Pro, 3.5Ghz, 6-core),
* Research problems require more compute power
* Soon be at a disadvantage if you don't


### Aim for Today

* Overview, 
    * hardware, machines, 
    * buzzwords
* Cluster Computing
* (next week, start parallel programming)

[MJonesTutorial]: http://www.buffalo.edu/content/www/ccr/support/training-resources/tutorials/advanced-topics--e-g--mpi--gpgpu--openmp--etc--/2011-01---introduction-to-hpc--hpc-1-/_jcr_content/par/download/file.res/introHPC-handout-2x2.pdf
[WikiPediaSuperComputer]: http://en.wikipedia.org/wiki/Supercomputer
[SutterWTTJ]: http://herbsutter.com/welcome-to-the-jungle
[AppleA8]: http://en.wikipedia.org/wiki/Apple_A8