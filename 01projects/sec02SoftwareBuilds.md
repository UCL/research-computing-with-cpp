---
title: Building research software
---

Estimated reading time: 25 minutes

## Introduction

In an introductory C++ class, you might have worked with small projects containing just a handful of files with C++ source code, and might have compiled and linked them together into an executable program.
As a high-level zeroth order oversimplification, this consists of `compiling` (translating) each C++ source file in your project into a machine-specific `object` file, and then connecting up the symbols in them (`linking`) to resolve each symbol to a unique definition.
At the end of this process, we obtain an executable program that is intended to be run on that particular computer.
It should be noted that the symbol names (e.g. for variables and functions) in the source code are not typically retained in their original form (i.e. they become *mangled*) when processed by this toolchain (unless explicitly instructed to retain them in original internal reference tables).

If we encounter any error(s) during the compilation or linking phase, we investigate the probable cause in our source files, edit them and repeat the aforesaid process until we successfully obtain an executable program.
When we obtain an executable program, a rudimentary verification step could be to run the program and test its output against expected outputs for *a priori* chosen set of inputs.
When logical flaws (bugs) are discovered in our source files, we edit them to address these flaws (known as `debugging`).
It is typical to adopt this classical *edit-compile-debug* cycle by working at either a command-line interface or through appropriately labelled buttons from an Integrated Development Environment (IDE), which invoke the relevant compilation, linking and debugging software behind the scenes, and present a simplified graphical user interface to the developer.

## Scalability considerations for large projects

### Multiple types of executable outputs

While this above methodology suffices for small projects and enables one to focus on learning and developing a solid understanding of the programming language itself, this approach does not scale well to meet the needs of larger, more complex projects.
For instance, most C++ research projects will need to produce multiple kinds of executables concurrently, e.g. a small, performance-optimised binary that can run as efficient as possible on the target hardware, and a separate executable that contains references to all the original symbols embedded within the executable program (usually for debugging purposes by the developers).

### Cross-compilation needs

Secondly, programmers may write code on their laptops which typically have an x86_64 or ARM processor, and may wish to produce executable programs targeting multiple CPU architectures (e.g. different kinds of ARM/MIPS processors and/or advanced x86_64 processors with advanced vector instructions typically found research cluster installations).
This is known as *cross-compilation*.
This means that the compilation/linking process should now involve queries of the underlying hardware to identify and enable hardware-specific capabilities into the final executable(s).

### Cross-platform support

The C++ standards (i.e. formal documents specifying the details of the C++ programming language) are agnostic of the target Operating System i.e. in theory at least C++ code written using only the core language and the standard library should be able to run on any OS for which a conforming toolchain exists, and library writers might wish to provide cross-platform support to users irrespective of the OS.

### External dependencies

Large scale C++ research projects are dependent on many third-party external libraries that are either pre-installed or are compiled along with the project.
For instance, the finite element library [Deal.II](https://dealii.org/), which won the Wilkinson award in 2007 for numerical software, relies on 37 other core and optional third-party libraries (for meshing, parallelisation, dense and sparse matrices, graph partitioning etc) and each of these dependencies further depend on dozens of other libraries and so on, resulting in a combinatorial explosion.
All high quality scientific software use third-party testing libraries for code verification.
Several scientific libraries run these checks in automated pipelines upon each edit to the source code, which enables larger and geographically distributed developers to contribute high-quality code while trying to avoid known bugs or failures.


### Multiple programming languages

Furthermore, large scale research software tend to use a heterogeneous mix of programming languages.
For instance, the Fenics finite element library consists of several components, with various [performance-critical components](https://github.com/FEniCS/dolfinx) written in C++ and other user-facing components written in domain-specific variants of scripting languages such as Python.
Other research software might contain a mix of Fortran and C++ e.g. if a decades-old scientific project is transitioning from Fortran to C++.
It is increasingly common to have GPU computing enabled (conditionally, if available) in modern massively parallel computational operations which may additionally involve the CUDA language.
The most common form of distributed memory parallelism in high performance computing environments is a library written in plain C.
This heterogeneous nature of research projects imply the correct invocation of multiple toolchains to produce the final executable output.

The need to invoke multiple toolchains, conditionally enabling various capabilities, that are dependent on hardware inspection, for each platform and OS quickly becomes non-scalable.
Any such executable (or in general any output artefact of the workflow) produced with specific desired properties is considered as a *software build* (or simply a *build*).
It is evident that obtaining software builds through manual invocation of various tools either at the command-line or through basic IDE options quickly becomes cumbersome and tedious.

### Build automation tools

#### Makefiles

The complexities of build tools have been acknowledged since nearly half a century, and several build automation tools (also known as build systems) have been devised over time to tackle the problem.
An early such tool, which is still popular today, is [`make`](https://en.wikipedia.org/wiki/Make_(software)).
In this build system, the library author writes a set of plain-text files, known as MakeFiles, and provides them along with the C++ source code.
For system inspection, a set of companion tools called [GNU Autotools](https://en.wikipedia.org/wiki/GNU_Autotools) was developed to configure the build capabilities depending on the target processor's capabilities.

While this works quite well (even today) for a large number of scenarios, the `Make` and `Autotools` system have a number of drawbacks by modern yardsticks.
For instance, they are available only in Unix-like OSes. MakeFiles use various cryptic shorthand notations and are frustratingly stubborn about formatting requirements.
The syntax makes is hard to fathom the relationships between various dependencies, and there are no built-in mechanisms to visualise them.
There is no built-in debugger.
In short, it is tedious to write MakeFiles by hand for large projects, but it is a good build system, as long as the required MakeFiles can somehow be generated for us.
Enter build system generators!

#### Build system generators

Non-Unix build systems have their own native build systems.
For instance on Windows, with the Microsoft C++ compiler and toolchain (MSVC), the build system is known as Visual Studio (an IDE exclusive to Windows) solutions, and on macOSes, the toolchain provided by Apple produces files in a proprietary format that can be read in by Xcode (an IDE exclusive to macOS) and then compiled to produce the final executable.

In late 1999/early 2000, there was no compatible build system that worked across all Operating Systems, and developers had to go through the frustrating process of manually invoking multiple toolchains on different systems.
It was during this time that, Kitware Inc, a company that was part of a consortium tasked with designing the C++-based Imaging Toolkit (itk) funded by a US national research grant, developed CMake, a cross-platform *build system generator* (a system that emits output for consumption by native build systems, which in-turn have to be invoked to produce the final build output).

Since then CMake has continuously evolved and improved, and while there certainly remain some valid criticisms which have spawned numerous other alternatives, it has become the de-facto build system generator of choice in C++ (and heterogeneous language mix) projects, both in academia and in industry.
CMake can be used from the command-line while being able to generate outputs directly consumable by various IDEs.
On the complementary side, a number of modern IDEs have integrated support for CMake projects through their graphical user interfaces.
At present, the majority of scientific packages use CMake.
Most third-party libraries and dependencies provide some mechanism/tooling support for CMake to help configure and build our projects.
Given these considerations, in this course, we shall be adopting CMake for building our C++ projects.
