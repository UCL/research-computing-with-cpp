---
title: Micro-benchmarking
---

## Micro-benchmarking

meaningful bit of code, and systematically run performance tests for each
Scientific method applied to performance: measure the time taken by each
commit.

Profiling can tell us what part to include in a benchmark.

Possible micro-benchmarking frameworks:

- timers in unit test framework (not really accurate)
- [google/benchmark](https://github.com/google/benchmark)
- [hayai](https://github.com/nickbruun/hayai), a "google-test"-like framework
- [others](http://www.bfilipek.com/2016/01/micro-benchmarking-libraries-for-c.html)

Questions micro-benchmarking can answer:

1. How long does it take on average
1. Standard-deviation from average
1. Worst case

Questions it doesn't always answer:

1. Performance of large subsets or whole application
1. Parallelization: communication vs computation

## Exercise: build and run `micro_benchmark`

`micro_benchmark` reproduces the evaluation function from the travelling
salesman problem.

It reproduces how a micro-benchmark framework works:

1. run code N times for warm-up
1. run code N' times for actual measurement

Make sure the code is built in `Release` mode.

Questions:

1. Why is the float implementation faster/slower? Does the speed-up change with
   the number of dimensions or the number of cities? What happens for Nrow = 3?
1. Write a function that computes the distance manually (without Eigen syntactic
   sugar). It it faster for Nrows=2? What about Nrows=8?

Remark:

   None of these have tests (this is an exercise in bad code, after all). Do
   you trust we are solving the travelling salesman problem? I *know* that
   `awful` does not, beyond the memory bugs... Because I added a bug. But maybe
   there are further bugs still.

   How much easier would it be to test the `manual` code above if we had tests
   for the evaluation function?
