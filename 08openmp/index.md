---
title: "Week 8: Parallel Programming with OpenMP"
---

## Overview

This week we'll be focusing on speeding up our code by taking advantage of **parallelism**, i.e. splitting up problem and giving different pieces to different processors. In these notes we'll introduce the following concepts:

- What is parallel programming?
  - Why do we need parallel programming?
  - What are shared- and distributed-memory parallelism?
  - What are weak and strong scaling?
- An introduction to OpenMP
  - Parallelising loops
  - Reductions
  - Data sharing
  - Thread control
  - Schedules
  - Parallelising multi-dimensional loops

## What is parallel programming?

So far in this course we've been writing fundamentally **sequential** programs, where the code runs from start to finish, each step one after another. Our programs don't try and do multiple operations at once. However, modern **multi-core** CPUs are perfectly capable of performing multiple operations in **parallel**. Our sequential programs are like single queues at a supermarket checkout, capable of processing only one person's items at a time. Open more lanes and we can process a lot more people! If we can split up our problems appropriately and run each piece in parallel, we should be able to speed up our code immensely.

Let me take a moment to briefly discuss a short history of parallel *hardware* before we dive properly into writing parallel software. When transistor CPUs were first developed in the 70s, they were capable of performing only one instruction at a time, a single add, comparison, memory read, etc. A CPU like the very early Intel 8086 with a clock speed of 10 MHz could still perform on the order of tens of millions of instructions per second, but still only one operation at a time. In personal computing at least, this was the era of the **single-core** CPU. 

**Vectorised** CPUs developed around the same time can operate on multiple pieces of data with an appropriate instruction which is technically a form of parallelism, albeit restricted to performing the *exact* same instruction on many pieces of data. All the same, these machines were extremely powerful for the time and dominated supercomputer design until the 90s.

**Multi-threading** allowed single-core CPUs to *appear* as if they were performing operations in parallel, for example allowing a PC user to listen to music and write a document at the same time. This was done by running multiple **threads** of execution and rapidly switching between them, for example switching between the thread running the music player, another running the text editor, and others running various parts of the operating system. This is called **context switching**. If you look at the processes or tasks running on your own PC, you'll see many more processes running than available cores. The operating system and CPU are working in tandem to cleverly disguise their **context switching** and make it appear all these processes are running at once. Multi-threading is rarely needed in supercomputing and is mainly a feature for consumer machines.

In order to perform truly parallel computation, that is two completely different operations happening at the exact same moment, CPUs had to become **multi-core**, hosting many processing units within one CPU and using them to run many threads in parallel. In consumer PCs this transition started with the dual-core Intel Core2Duo series in the mid-2000s, but supercomputers were already playing with the slightly related idea of **multi-CPU** systems with hundreds of CPUs as early as the 80s! As of 2023, core counts on both server and consumer CPUs can be as high as 64. Typical consumer systems will have only one CPU while servers may have several on one motherboard. Writing parallel programs for multi-core machines is made easy using the focus of this week: **OpenMP**.

Up until the 90s, supercomputers were dominated by **shared-memory** systems, where all processing units accessed one single memory space. In the late-80s/early-90s it become economically more feasible to increase parallelism and computational power by creating **distributed** systems made up of multiple **nodes**. Each node is a single computer with its own CPU(s), memory space, networking hardware, and possibly other components like hard drives, all networked together in such a way they can communicate information and operate as *one single parallel machine*. I'll start using the term **node** to refer to a single computer as part of a larger, distributed parallel machine. While shared-memory systems still exist[^jq-terabytes], nearly all the world's largest computers are now distributed systems.

Distributed systems tend to have separate memory spaces for each node[^multi-cpu] which introduces a new complexity: how do the nodes share data or communicate when necessary? These **distributed-memory** systems are commonly programmed using a library called the **Message Passing Interface** (MPI) which allows individual nodes to communicate by passing data back and forth. We'll get into MPI properly in the last part of this course. For now, it's useful just to understand the difference between the two basic forms of parallel computing: **shared-memory** parallelism where each thread shares the same memory space and can access any piece of data within it, and **distributed-memory** systems where threads do not share one memory space and must explicitly send and receive data when necessary.

What I haven't touched on here is the incredible parallelism of **graphical processing units (GPUs)**. Since we won't teach GPU programming in this course, but it is still a very relevant skill for developing HPC codes in research, I want to mention that Nvidia provide a number of excellent introductory courses in programming GPUs with CUDA. See [their course catalogue](https://www.nvidia.com/en-us/training/) for more details.

[^jq-terabytes]: I (Jamie Quinn) used to program a modern machine with 4 *terabytes* of RAM!

[^multi-cpu]: In fact, systems with multiple CPUs (not cores) can still have separate memory spaces within a single node.

## Why do we need parallel programming?

We as researchers turn to parallel computing when our codes either are too slow (even with the best single-core performance) or take up too much memory to fit inside one system's memory space. For a quick motivating example, consider the fastest supercomputer in the world: [Frontier](https://en.wikipedia.org/wiki/Frontier_(supercomputer)). In total it has around 600,000 CPU cores, and several GPUs as well but let's just consider the CPU power here. Codes tend to only use a fraction of a large supercomputer like this at one time, so let's assume a code is only using 10%; 60,000 cores. If a code runs for a single day on this piece of Frontier, a single-core version of that same code could run for over 150 years. Now, parallel scaling (which we'll discuss in a later section) might not be perfect, so let us assume an unrealistically *terrible* parallel scaling of 1% over the single-core version, giving a single-core runtime of 1.5 years. More likely, parallel scaling is over 50%, and these kinds of codes rarely run for as short a time as a single day, some simulations taking over a month to complete. You can imagine the insignificant size of problems we could solve if we couldn't use parallel computing.

You might be asking, why not just make CPUs faster and process more data per second. Why do we even need parallelism at all? While CPU clock speeds would increase by nearly a thousand times between the first microprocessor and now, and clever tricks inside the CPU allows more instructions per second to be carried out, the fundamental physics of power and frequency in electronic components limit how much faster single-core CPUs can get. See [Wikipedia on Frequency Scaling](https://en.wikipedia.org/wiki/Frequency_scaling) for more details on the power barriers in CPU design.

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
- point to point communications

In essence, if there is a prescribed order to tasks, where one *must* complete before another, it is unlikely that they can be parallelised[^parareal].

[^parareal]: There are some deviously clever algorithms that can somewhat get around temporal ordering like the [Parareal algorithm](https://en.wikipedia.org/wiki/Parareal) but these are not widely applicable.

## An introduction to OpenMP

OpenMP is a way of parallelising C++ and Fortran code for multi-core, shared-memory systems. It can also offload computations to accelerators like GPUs but we won't go into that here. The way we use OpenMP is through **preprocessor directives**, statements that augment our code to give extra information to the compiler, allowing it to parallelise the code automatically. We'll describe the syntax for these directives in a later section but as a very quick example of what OpenMP looks like, this is how we can parallelise a loop using a `parallel for`:

```cpp
#pragma omp parallel for
for(...) {
  // perform some independent loop iterations
}
```

That's it! That's how simple OpenMP *can* be. Of course, OpenMP exposes much more functionality, allowing us to parallelise more complex code, and optimise *how* OpenMP parallelises our code. We can also use OpenMP to introduce some subtle and dangerous bugs (this is C++ after all) but at its core, OpenMP is a remarkably accessible way to program for shared-memory machines and has become standard in the HPC world for this purpose.

OpenMP is a deep and rich way to express parallelism within our code, and the directive-based approach allows us to support a version of our code compiled *without* OpenMP. **If we compile this code without informing the compiler we wish to use OpenMP, it will simply ignore the directives and treat it as sequential code.** If, instead, we use OpenMP's library interface, we have to work a little harder to ensure the code will still compile without OpenMP but it is still possible. Most of the HPC community use the directive-based approach, so that's what we'll focus on here.

I should also note that OpenMP has many more features than the basic functionality I'll discuss here. Much like C++ itself, OpenMP is not a library or a piece of software but a *specification* which is continually evolving, with the latest 5.2 version being released in Nov 2021. Compiler developers will implement parts of the specification as appropriate for their userbases and support in GCC at least is minimal for the very latest features. See the [OpenMP page in the GCC wiki](https://gcc.gnu.org/wiki/openmp) for details.

### Compiling OpenMP code

Telling the compiler to parallelise our code using OpenMP is as surprisingly simple as its basic usage. Let's look at how we can use OpenMP using just `g++` and as part of a CMake project.

#### Compiling with `g++`

If we're using `g++` to compile our program like `g++ -o hello hello.cpp` then all we need to add is the OpenMP flag:

```bash
g++ -fopenmp -o hello hello.cpp
```

Other compilers have similar flags and, as previously mentioned, support different parts of OpenMP, although all compilers should support the basic features like the `parallel for` we've seen in the earlier example.

#### Compiling with CMake

Stealing this snippet from the excellent[Modern CMake](https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html), we can add OpenMP to a CMake project (using CMake 3.9+) with:

```cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MyTarget PUBLIC OpenMP::OpenMP_CXX)
endif()
```

### What is `#pragma omp`?

You will likely only interact with OpenMP through its directives and every directive starts the same way:

```cpp
#pragma omp ...
```

You have already seen a similar kind of preprocessor directive when including header files! `#include <iostream>` tells the C++ **preprocessor**[^preprocessor] to paste the contents of the iostream header directly into the C++ file to make its contents available to that file. When writing your own header files, you may have also come across a pragma directive: `#pragma once`. This tells the preprocessor to *only* include the header file if it has *not* already been included somewhere earlier in the chain of included files, particularly if it was already included inside another header file. If you've seen older header files, you may have also come across `#define` and `#if` to create **include guards** around the contents of header files, which allow the preprocessor to similarly avoid including a header multiple times in one file. 

[^preprocessor]: You may not have had to learn much about the preprocessor while using modern C++. It is used far more widely in older C++ and in C and, really, modern C++ has better built-in ways to provide the same functionality the preprocessor provides. At a high level, it is a text processor that can understand specific directives like `#include` and manipulate the text in C++ files appropriately. In C++ development you likely won't need to know much more than this.

OpenMP uses these `#pragma` directives to describe what pieces of code should be automatically parallelised and, if necessary for optimisation, how the parallelisation should happen. If this is a little confusing at this stage, don't worry, it will become clear as you start playing with the OpenMP constructs through some examples.

### Parallelising a loop

We've already seen an example of parallelising a loop:

```cpp
#pragma omp parallel for
for(...) {
  // perform some independent loop iterations
}
```

This **construct**[^construct] actually combines two ideas:

- creating a `parallel` **region** to spawn some **worker threads**
- splitting up a loop's work between the threads

[^construct]: In OpenMP **construct** is the term for a region of code that OpenMP acts on, including the directive, so here we are using the `parallel for` construct. We'll see a few more OpenMP constructs as we got through more examples.

This `parallel for` is actually shorthand for two separate constructs which we can explicitly write as:

```cpp
#pragma omp parallel
{
  #pragma omp for
  for(...) {
    // perform some independent loop iterations
  }
}
```

Notice the new set of curly brackets which defines a new **scope**. At first this may seem strange, but scope is how OpenMP knows when to end the parallel region (or any other construct that applies to a *region* of code). You should already know about scope from your previous C++ knowledge so I won't explain it here. In C++ loops create their own new scope inside the loop delineated by curly braces, hence why we don't have to add any new braces when using the `parallel for`; it only applies to the duration of the for loop. But when using only the `parallel` construct, we have to explicitly tell the compiler where the parallel region starts and ends.

The `parallel` directive tells OpenMP that we want to parallelise the piece of code inside the following scope. What this really means is when execution reaches this parallel region, OpenMP will create a number of threads, allow any constructs inside the parallel region to use those threads, and then destroy those same threads. Strictly, the `parallel` construct *does nothing by itself* except create and destroy threads.

The `for` construct in this example uses the threads spawned by `parallel` by splitting the iterations of the for loop into chunks and assigning different chunks to different threads. Default behaviour can change based on compiler and OpenMP runtime but usually the `for` construct will split the loop into as many chunks as their are threads and assign approximately the same number of iterations to each thread. So if there are 16 loop iterations to be performed at 4 threads, you can probably expect one thread to take iterations 0-4, another to take 5-8, and so on. Not all loops are this simple so we'll discuss later how we can optimise the `for` construct by specifying how the iterations are split amongst the threads.

### Example 1: Filling an array

Let's apply this `parallel for` to an example where we want to fill an array with values from a moderately expensive function like `sin`:

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "timer.hpp"

using namespace std;

int main() {
  const int N = 100'000'000;
  vector<double> vec(N);

  Timer timer;

#pragma omp parallel for
  for(int i=0; i<vec.size(); ++i) {
    vec[i] = sin(M_PI*float(i)/N);
  }

  std::cout << timer.elapsed() << '\n';

  return 0;
}
```

You can find this example and all further examples in the examples repository TODO. The timer code in the example is relatively simple so I won't explain it in detail, it just starts a timer when it's created and shows the elapsed time when `timer.elapsed()` is called.

Let's build this *without* OpenMP to

1. show that the compiler will simply ignore the pragmas if we don't explicitly tell it to use OpenMP and
2. measure the time of the serial code.

Building this with `g++ main.cpp timer.cpp` and running with `./a.out` prints out the time taken to fill this vector. It should take less than a second or so on your machine. Running on my laptop once, I got a time of $0.98$s. Adding the OpenMP flag for GCC, compiling with `g++ -fopenmp main.cpp timer.cpp` and again running with `./a.out` gives me a time of $0.14$. For me that's a speed up of around $700$% with one additional line of code! Given I have an 8-core processor, $800$% would be ideal but perfect scaling isn't usually possible in parallel computations, and perhaps with some optimisation we could squeeze out a little more performance. What performance increase do you see when running on your machine and does it match what you expect from the number of cores in your CPU?

I should point out here that this performance is when using the default compiler optimisation level `-O0`. If we switch to using `-O2` with `g++ -fopenmp -O2 main.cpp timer.cpp`, the serial performance drops to $0.74$s and the parallel performance to $0.10$, now giving a speedup of $740$%. While parallelisation can improve our overall performance greatly, single-core optimisations are still crucial to achieving the best performance possible from a code.

### Example 2: Summing an array (naively)

Let's now sum the array from the previous example. I won't go into the maths but the sum of this actually gives us a numerical approximation of the integral of $sin(\pi x)$ from $0$ to $\pi$. If we sum the array, multiply it by $\pi$ and divide by $N$ we should get *exactly* $2$. The filling of the vector is the same so I'll only print here the code for the sum itself:

```
  double sum = 0.;

  Timer timer;

#pragma omp parallel for
  for(int i=0; i<vec.size(); ++i) {
    sum += vec[i];
  }

  double elapsed = timer.elapsed();

  std::cout << "Time: " << elapsed << '\n';
  std::cout << "Result: " << M_PI*sum/N << '\n';
```

Compiling and running without OpenMP gives:

```
Time: 0.276116
Result: 2
```

This is exactly what we expect. If you enable OpenMP however, you'll probably get something like:

```
Time: 0.957646
Result: 0.14947
```

I recommend you take a break from reading/coding here and think about the questions, *why has the time increased with more threads* and *why is the answer wrong*? Don't worry if it's not obvious, this is a subtle and tricky bug and requires thinking about what each thread is doing as it accesses each variable inside the loop.

---

This behaviour is the result of a **data race**. Consider even just two threads working in parallel inside the loop. Thread 1 reads data from `vec[i]`. The `for` construct has split the loop between the two threads so its value of `i` is totally unique to thread 1 and it is the only thread touching the data in `vec[i]`. It then needs to add this value into `sum`. It reads `sum`, adds `vec[i]` to it and writes it back into `sum`. This addition takes some time however, and thread 2 is performing the exact same operation but using a different value of `i`. When thread 1 reads `sum` and is currently spending time adding, thread 2 could easily have just finished an addition and has written a new value to `sum`. Thread 1 has no idea that thread 2 has done this and then unknowingly *overwrites* the updated value of `sum` with its own version. Not only does this mean the value of `sum` is *incorrect* but threads are not allowed to write to the same variable at the same time, so each threads must wait for the other to finish, adding more time to the overall calculation. This is why the overall time has increased, despite using multiple threads, and why the result is wrong!

So how do we handle a data race? Each example of a data race is unique and must be handled in its own way, but the general solution is to limit threads to writing to variables no other threads will write to. We'll look later at a few different ways to tackle this particular problem but summing is part of a larger group of operations called **reductions**, where a large data structure (like a vector) is reduced to a smaller number of values, in this case just a single number. Thankfully, OpenMP provides a way to easily deal with these kinds of operations: reduction clauses.

### Example 3: Summing an array with `reduction`

OpenMP allows constructs to be augmented with **clauses**, extra pieces of information we can add to its directives to further refine the parallelisation. In the above sum example, we can tell OpenMP we're trying to perform a reduction by adding the `reduction` clause to the `parallel for` construct along with the operator we want to reduce using and the variable that is holding the value of the reduction:

```
  double sum = 0.;

  Timer timer;

#pragma omp parallel for reduction(+:sum)
  for(int i=0; i<vec.size(); ++i) {
    sum += vec[i];
  }

  double elapsed = timer.elapsed();

  std::cout << "Time: " << elapsed << '\n';
  std::cout << "Result: " << M_PI*sum/N << '\n';
```

If you run the above example with OpenMP you should get something like:

```
Time: 0.0667325
Result: 2
```



### How to specify the number of threads in OpenMP?



### Fractal generation

[Fractals](https://en.wikipedia.org/wiki/Fractal) are shapes or patterns that in some sense contain infinite complexity. I won't go into their (very interesting) mathematical properties but plotting a fractal like the classic Mandelbrot set can produce some beautiful patterns:

![](img/mandelbrot.jpg)

The above plot is made by turning each pixel with coordinates $(p,q)$ into a complex number $c = p + iq$ where $i^2 = -1$ is the imaginary unit. $c$ is used in the iterative equation $z_{n+1} = z_n^2 + c$ starting from $z_0 = 0$ and $z_n$ updated until either $|z_n| > 2$ or some maximum number of iterations is reached. It's not crucial to understand the maths behind this, 

## Numerically computing an integral

We can use common numerical techniques to turn an integral into a sum and compute than sum using a computer. Take calculating $\pi$ with the integral:

$\pi = 4\int_{0}^{1} \frac{dx}{1+x^2}$

Using the [midpoint rule](https://en.wikipedia.org/wiki/Numerical_integration#Quadrature_rules_based_on_interpolating_functions) we can turn this into a sum over $N$ pieces:

$\pi \approx \frac{1}{N}\sum_{i=1}{N} \frac{1}{1 + \left( \frac{i-1/2}{N} \right)^2}$

It's not too important to be able to derive this sum yourself if it's not your background, but you should be able to implement it in code. We could write the sum in C++ like:

```cpp
int N = 1024;
double sum = 0.0;
for (int i=1; i<N; ++i) {
  sum += 1.0 / ( 1.0 + pow((float(i) - 0.5)/N, 2.0) );
}
double pi = 4.0 * sum / N;
```

$N=1024$ will run in a tiny amount of time on a modern CPU but if our integrand were more complex, if this were a 2D integral, if $N$ was extremely large, this sum could take long enough that it's worth parallelising. But how might we do that? The key here is noticing that the sum can be performed in any order, it fits our requirement of a parallel problem that pieces of the computation can be performed independently.

If I asked you personally to perform this sum, you might be able to do it in a few hours, but if you gave the first quarter of the sum (the first 256 pieces) to a friend to add up, the second quarter to another friend, the third to a third, and you do the last quarter yourself, then add up all your sums, you're going to get the final answer much faster than individually. You'll be limited by the slowest adder but you'll still get your answer quicker. This is the basic idea of parallelising a sum, or really any loop where each loop iteration is independent. You might argue that each loop iteration isn't truly independent because all iterations add to a single sum but 
