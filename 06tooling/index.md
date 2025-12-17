---
title: "Week 6: Libraries and Tooling"
---

This week we'll learn about some of the tools that we can use to improve our code, and how to link with external libraries. 

You should look over and install the following tools, and familiarise yourself a little with the timing statements available in C++. 

1. [Timing and Tooling](sec00TimingAndTooling.html)

## Why use Libraries?

Many programming tasks are shared between a wide variety of projects, especially within particular communities and fields. Reusing code that has already been written not only saves precious time, but also allows us to use code that has been widely and thoroughly tested. A large community of people using a library can also help to ensure that bugs in that library are spotted and reported. 

### What are libraries?

- Libraries are collections of useful classes and functions, ready to use
- C++ libraries can be somewhat harder to use than modules in other languages (e.g. Python)
- Can save time and effort by providing well-tested, flexible, optimised features

### Libraries from a scientific coding perspective

Some things that scientists in research generally ought not to build themselves:

- standard data structures (most of these are actually provided for us in the C++ standard library),
- file input/output for standard file formats,
- common but non-trivial numerical algorithms (e.g. sorting, linear solvers, FFT, etc),
- graphical outputs like plots, gui and so on.

Sometimes we have to build things ourselves, when:

- a library isn't fast enough,
- we don't trust a library's results/methods,
- a library doesn't provide the needed functionality,
- we can't use a library due to licensing issues,

It's also worth bearing in mind that having multiple implementations can be important for science. Bugs exist in almost all software and can be very subtle, and numerical inaccuracies can be even subtler. If everyone in a field uses the same library for some key scientific calculation then problems with that library may be very difficult to spot: having multiple libraries in a scientific community can be useful as independent checks of one anothers results. 

1. [Choosing Libraries](sec01ChoosingLibraries.html)
2. [Library Basics](sec02LibraryBasics.html)
3. [Linking Libraries](sec03LinkingLibraries.html)
4. [Installing Libraries](sec04InstallingLibraries.html)
5. [Libraries Summary](sec05Summary.html)
