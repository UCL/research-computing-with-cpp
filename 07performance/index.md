---
title: "Week 7: Introduction to Performance"
---

## Week 3: Overview 

This week we'll be introducing some concepts for producing high performance programming on a single CPU core, before moving on to parallel programming in the next three weeks. 

Even though parallelism can help us improve our throughput, single core optimisation is still vital for producing good performance on ordinary machine or for maximising the work that each core can do in a parallel program. This week we'll talk about:

1. [_Why_ and _when_ we should optimise](sec00Motivation)
2. [Complexity and algorithm analysis](sec01Complexity) 
    - How the time and space usage of algorithms scales with the size of the input. 
    - Big-O notation.
    - How to determine complexity and examples with some common algorithms.
    - How does complexity impact our choices as software designers?
3. [Memory management and caching](sec02Memory)
    - Memory bound problems. 
    - How memory is structured in a typical machine.
    - Speed of different kinds of memory access.
    - Cache structure and operation. 
    - Writing algorithms to effectively exploit the cache. 
4. [Compiler Optimisation](sec03Optimisation)
    - Automated optimisation by the compiler. 
    - Compiler flags for optimisation.
    - Examples of optimisations, pros and cons.
    - Working with optimisation in practice. 

## Recommended Texts for Further Reading

- **Introduction to Algorithms**, Cormen, Leiserson, Rivest, and Stein. Covers many different kinds of algorithms including those mentioned in the notes this week, and gives an introduction to algorithmic analysis, big-O notation, and applies this analysis to algorithms throughout. 
- **Computational Complexity** 
- **Handbook of Floating Point Arithmetic**, Muller et al. 
- **The Art of Programming**, Donald Knuth. A very thorough (if somewhat dense) discussion of floating point arithmetic. 