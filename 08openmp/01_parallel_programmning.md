---
title: What is parallel programming?
---

## What is parallel programming?

So far in this course we've been writing mostly **sequential** programs, where the code runs from start to finish, each step one after another. Last week you were briefly introduced to SIMD instructions (single instruction, multiple data) where CPUs can optimise the application of a single instruction to multiple pieces of data when appropriate. This week we move into the world of MIMD (multiple instructions, multiple data). Modern **multi-core** CPUs are perfectly capable of performing many *different* operations completely independently of one another, i.e. in **parallel**. Our sequential programs are like single queues at a supermarket checkout, capable of processing only one person's items at a time. Open more lanes and we can process a lot more people! If we can split up a problem into pieces and run each piece in parallel, we can achieve impressive performance improvements.

## A brief history of parallel hardware

Let me take a moment to briefly discuss a short history of parallel *hardware* before we dive properly into writing parallel software. When transistor CPUs were first developed in the 1970s, they were capable of performing only one instruction at a time; a single add, comparison, memory read, etc. A CPU like the very early Intel 8086 with a clock speed of 10 MHz could still perform on the order of tens of millions of instructions per second, but still only one operation at a time. In personal computing at least, this was the era of the **single-core** CPU. 

**Vectorised** CPUs developed around the same time can operate on multiple pieces of data with SIMD instructions which are technically a form of parallelism, albeit restricted to performing the *exact* same instruction on many pieces of data. All the same, these machines were extremely powerful for the time and dominated supercomputer design until the 90s.

**Multi-threading** allowed single-core CPUs to *appear* as if they were performing operations in parallel, for example allowing a PC user to listen to music and write a document at the same time. This was done by running multiple **threads** of execution and rapidly switching between them, for example switching between the thread running the music player, another running the text editor, and others running various parts of the operating system. This is called **context switching**. If you look at the processes or tasks running on your own PC, you'll see many more processes running than available cores. The operating system and CPU are working in tandem to cleverly disguise their **context switching** and make it appear all these processes are running at once. Multi-threading is rarely needed in supercomputing and is mainly a feature for consumer machines. In HPC we generally use one thread per core.

In order to perform truly parallel computation, that is two completely different operations happening at the exact same moment, CPUs had to become **multi-core**, hosting many processing units within one CPU and using them to run many threads in parallel. In consumer PCs this transition started with the dual-core Intel Core2Duo series in the mid-2000s, but supercomputers were already playing with the slightly related idea of **multi-CPU** systems with hundreds of CPUs as early as the 80s! As of 2023, core counts on both server and consumer CPUs can be as high as 64. Typical consumer systems will have only one CPU while servers may have several on one motherboard. Writing parallel programs for multi-core machines is made (relatively) easy using the focus of this week: **OpenMP**.

Up until the 90s, supercomputers were dominated by **shared-memory** systems, where all processing units accessed one single memory space. In the late-80s/early-90s it became economically more feasible to increase parallelism and computational power by creating **distributed** systems made up of multiple **nodes**. Each node is a single computer with its own CPU(s), memory space, networking hardware, and possibly other components like hard drives, all networked together in such a way they can communicate information and operate as *one single parallel machine*. I'll start using the term **node** to refer to a single computer as part of a larger, distributed parallel machine. While shared-memory systems still exist[^jq-terabytes], nearly all the world's largest computers are now distributed systems.

[^jq-terabytes]: I (Jamie Quinn) used to program a modern machine with 4 *terabytes* of RAM!

Distributed systems tend to have separate memory spaces for each node[^multi-cpu] which introduces a new complexity: how do the nodes share data or communicate when necessary? These **distributed-memory** systems are commonly programmed using a library called the **Message Passing Interface** (MPI) which allows individual nodes to communicate by passing data back and forth[^mpi]. We'll get into MPI properly in the last part of this course. For now, it's useful just to understand the difference between the two basic forms of parallel computing: **shared-memory** parallelism where all threads share the same memory space and can access any piece of data within it, and **distributed-memory** systems where threads do not share one memory space and must explicitly send and receive data when necessary.

[^multi-cpu]: In fact, systems with multiple CPUs (not cores) can still have separate memory spaces within a single node.

[^mpi]: Technically MPI is a standard which is implemented by libraries like OpenMPI (not to be confused with OpenMP) and MPICH. Vendors like Cray, Intel, IBM and Nvidia often provide their own MPI implementations specifically tuned to clusters they support.

What I haven't touched on here is the incredible parallelism of **accelerators** like **graphical processing units (GPUs)**. Since we won't teach GPU programming in this course, but it is still a very relevant skill for developing HPC codes in research, I want to mention that Nvidia provide a number of excellent introductory courses in programming GPUs with CUDA. See [their course catalogue](https://www.nvidia.com/en-us/training/) for more details.


## Why do we need parallel programming?

We as researchers turn to parallel computing when our codes either are too slow (even with the best single-core performance) or take up too much memory to fit inside one system's memory space. For a quick motivating example, consider the fastest supercomputer in the world: [Frontier](https://en.wikipedia.org/wiki/Frontier_(supercomputer)). In total it has around 600,000 CPU cores, and several GPUs as well but let's just consider the CPU power here. Codes tend to only use a fraction of a large supercomputer like this at one time, so let's assume a code is only using 10%: 60,000 cores. If a code runs for a single day on this piece of Frontier, a version of that same code where all computation is done on a single core could run for over 150 years. These kinds of codes rarely run for as short a time as a single day, some simulations taking over a month to complete. You can imagine the insignificant size of problems we could solve if we couldn't use parallel computing.

You might be asking, why not just make CPUs faster and process more data per second. Why do we even need parallelism at all? While CPU clock speeds would increase by nearly a thousand times between the first microprocessor (in the 1970s) and now, and clever tricks inside the CPU allows even more instructions per second to be carried out, the fundamental physics of power and frequency in electronic components limit the processing speed of single-core CPUs. See [Wikipedia on Frequency Scaling](https://en.wikipedia.org/wiki/Frequency_scaling) for more details on the power barriers in CPU design.

## What problems can be parallelised?

Let's take a moment now to discuss the kinds of problems that *can* be parallelised. Fundamentally, **only problems that can be split into individual chunks and *processed independently* can be parallelised**. If one piece of computation depends on another, it *cannot* happen in parallel. Some examples of parallel tasks include:

- processing individual cells in a grid-based simulation
- simulating non-interacting particles
- processing orders in a restaurant
- searching for strings in a piece of text
- Monte-Carlo simulations

However, some tasks simply cannot be parallelised:

- all steps in baking a cake
- reading a text in order
- synchronous point to point communications

In essence, if there is a prescribed order to tasks, where one *must* complete before another, it is unlikely that they can be parallelised[^parareal].

[^parareal]: There are some deviously clever algorithms that can somewhat get around temporal ordering like the [Parareal algorithm](https://en.wikipedia.org/wiki/Parareal) but these are not widely applicable.

