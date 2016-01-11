---
title: C++ in Research Code 
---

## Using C++ in Research

### Some challenges

* In Science, people often learn MATLAB, Python, R first. 
    * C++ perceived as difficult, error prone, wordy. 

* C++ is a compiled language.
    * Compilers: Installation, versions, environment can be confusing.
    * Build infrastructure: compile-commands, managing dependent libraries can be complicated. 
    * Result: More difficult than interpretted languages.

* Often, you are not making a finished product.
    * Algorithms change.
    * Run algorithm via GUI, command line, cluster, web etc.
    * Result: Takes too long to adapt code to new uses.

* Result: It's more trouble than its worth?

### Some advantages

* Faster code than interpretted code.
    * Closer to the hardware.
    * i.e. Code compiles down to hardware specific instructions.

* Better integration with accelerated code.
    * e.g. CUDA, OpenCL, OpenACC
    * Other languages can access these, but its less clean.
    * i.e. fewer different technologies to learn.

* Many existing libraries are written in C++.
    * You want to re-use them.
    * So, you end up using it too.

* In research, you are often pushing the limits.
    * Large amounts of data.
    * Large amounts of compute time.
    * So, interpretted languages can become unworkable, and in effect just too slow.
 
* Result: Good reasons to use it, so lets learn to work with it.

### What Isn't This Course?

We are NOT suggesting that:

* C++ is the solution to all problems. 
* You should write all parts of your code in C++.

### What Is This Course?

We aim to:

* Improve your C++ (and associated technologies).
* Do High Performance Computing (HPC).
* Apply it to research in a pragmatic fashion.
* You need the right tool for the job.

