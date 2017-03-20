---
title: Post-coding medley, memory leaks, and performance measurements
---

## Side-note: Flynn's Taxonomy of Parallelization

- SISD: Single instruction single data

  prototypical serial code

- SIMD: Single instruction multiple data

  Same instruction is performed in parallel over different inputs. Necessary
  in GPU (at the level of a warp of 32 threads). Likely in OpenMP
  (for loop parallelization) and MPI.

- MIMD: Multiple instruction multiple data

  Basically, different threads or different nodes doing different things, e.g.
  computing different terms in an equation, dealing one with the GUI, the other
  with a database, etc...

- MISD: Multiple instructions single data

  Weird... Used for fault tolerance (different algorithm that should lead to
  same output).

## All tests passe, the code works: are we done yet?

Lots of changes can be made to a code, from cosmetic to crucial:

- formatting, linting, and refactoring
- checking for memory leaks
- profiling and performance
- benchmarking
