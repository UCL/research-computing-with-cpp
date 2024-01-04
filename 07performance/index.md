---
title: "Week 7: Introduction to Performance"
---

## Week 7: Overview 

This week we'll be introducing some concepts for producing high performance programming on a single CPU core, before moving on to parallel programming in the next three weeks. 

Even though parallelism can help us improve our throughput, single core optimisation is still vital for producing good performance on ordinary machines or for maximising the work that each core can do in a parallel program. This week we'll talk about:

1. [_Why_ and _when_ we should optimise](sec00Motivation.html)
2. [Complexity and algorithm analysis](sec01Complexity.html) 
    - How the time and space usage of algorithms scales with the size of the input. 
    - Big-O notation.
    - How to determine complexity and examples with some common algorithms.
    - How does complexity impact our choices as software designers?
3. [Memory management and caching](sec02Memory.html)
    - Memory bound problems. 
    - How memory is structured in a typical machine.
    - Speed of different kinds of memory access.
    - Cache structure and operation. 
    - Writing algorithms to effectively exploit the cache. 
4. [Compiler Optimisation](sec03Optimisation.html)
    - Automated optimisation by the compiler. 
    - Compiler flags for optimisation.
    - Examples of optimisations, pros and cons.
    - Working with optimisation in practice. 

## Recommended Texts for Further Reading

- **Introduction to Algorithms**, Cormen, Leiserson, Rivest, and Stein. 
    - Covers many different kinds of algorithms including those mentioned in the notes this week, and gives an introduction to algorithmic analysis, big-O notation, and applies this analysis to algorithms throughout. 
- **Computational Complexity, A Modern Approach**, Sanjeev Arora and Boaz Barak. 
    - A very accessible but more computer science oriented approach to complexity as a field; it goes into detail about things like complexity classes and relating complexity to the Turing Machine model of computation. Useful if you're interested in the bigger picture of computational complexity and why it is important. 
- **Handbook of Floating Point Arithmetic**, Muller et al.
    - Covers just about everything you could want to know about floating point arithmetic including approaches to fast and accurate numerical algorithms, compilers, hardware and software implementation, and the evolution of floating point standards! 
- **The Art of Computer Programming**, Donald Knuth.
    - The big famous one. A very thorough (if somewhat dense) discussion of computing from a low level point of view. Great if you want to approach these issues with rigour and a strong relationship to machine implementation - code examples are in a form of assembly. Book 1 (Fundamental Algorithms) and Book 3 (Sorting and Searching) detail algorithm implementations and precise discussions of algorithm performance. Book 2 (semi-numerical algorithms) has a good discussion of floating point arithmetic, its implementation, and errors. 
