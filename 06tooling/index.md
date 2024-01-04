---
title: "Week 6: Libraries and Tooling"
---

This week we'll learn about some of the tools that we can use to improve our code, and how to link with external libraries. 

You should look over and install the following tools, and familiarise yourself a little with the timing statements available in C++. 

1. [Timing and Tooling](sec00TimingAndTooling.html)

## Why use Libraries?

> The best code is the code you never write

### What are libraries?

- Libraries are collections of useful classes and functions, ready to use
- C++ libraries can be somewhat harder to use than modules in other languages (e.g. Python)
- Can save time and effort by providing well-tested, flexible, optimised features

### Libraries from a scientific coding perspective

Libraries help us do science faster

- Write less code (probably)
- Write better tested code (probably)
- Write faster code (possibly)

Particular things we scientists don't ever want to build ourselves:

- standard data structures (e.g. arrays, trees, linked lists, etc)
- file input/output (both for config files and output files)
- standard numerical algorithms (e.g. sorting, linear solve, FFT, etc)
- data analysis and plotting

Sometimes we have to build things ourselves, when:

- a library isn't fast enough
- we don't trust a library's results/methods
- a library doesn't provide the needed functionality
- we can't use a library due to licensing issues

1. [Choosing Libraries](sec01ChoosingLibraries.html)
2. [Library Basics](sec02LibraryBasics.html)
3. [Linking Libraries](sec03LinkingLibraries.html)
4. [Installing Libraries](sec04InstallingLibraries.html)
5. [Libraries Summary](sec05Summary.html)
