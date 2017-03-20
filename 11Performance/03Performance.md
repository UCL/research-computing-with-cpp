---
title: Profiling and measuring performance
---

## Amdahl's law

Give's the maximum theoretical speedup for parallelizing a given task.

- Given a program that takes a time T to execute
- When parallelizing a task A taking time P over n threads
- Then the maximum speedup is:

      $S = \frac{T}{T - P + frac{P}{n}}$

Some simple cases:

- $P \mapst 0$, $n \mapsto \infty$, then  $S \rightarrow 1$
- $P=0.5$, $n \mapsto \infty$, then  $S \rightarrow 2$

In practice, it means programmers/researchers should measure performance before
jumping to "optimize" code (a.k.a. apply the scientific method?).

## Profiling

Refers to measuring how much time the program spends in each function:

- Valgrind, via [callgrind](http://valgrind.org/docs/manual/cl-manual.html) provides an accurate, non-intrusive solution
- [gprof](https://sourceware.org/binutils/docs/gprof/) require "instrumenting"
  the program, e.g. recompiling with the gprof library. It polls the program
  every so often, a.k.a. "sampling"
- [XCode instruments](https://developer.apple.com/library/content/documentation/DeveloperTools/Conceptual/InstrumentsUserGuide/) provides "sampling" without requiring instrumentation

Other considerations to take into account: threads, GPU-specifics, MPI...

## Exercise: Profiling the correct `less_bad`

1. install `kcachegrind` or `qcachegrind` (latter on Mac + Homebrew, or
   Windows)
1. recompile in release mode + debug info `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo`
1. Run the following command:

```
> docker run --rm \
    -v /path/to/source/on/vm:/path/to/source/on/container \
    -w /path/to/source/on/container  \
    course_container \
    valgrind -v --tool=callgrind ./awful
```

Questions:

1. How much time is spent on `neighbor`
1. How much time is spent on `TravelDistance::operator()`
1. What happens when you increase the number of cities to 100, or 10000
