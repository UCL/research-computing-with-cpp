---
title: "Week 6: Libraries and Tooling"
---

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


This week we'll also introduce a number of tools which we can use to develop and improve our code. We'll start using these in the practical but please make sure you install and set up the tools ahead of time, and reach out to us before class if you have trouble doing so.

**N.B.** Please remember that if you are using Windows for this course you will need to install these tools **inside WSL** (Windows Subsystem for Linux) rather than following a normal Windows installation. To do so, you can 
1. Open a Command Prompt and type `wsl` to go into the WSL command line. From there you can follow Linux instructions for installing command line tools like Valgrind. 
2. Open VSCode and [connect to WSL using the button in the bottom left hand corner](https://code.visualstudio.com/docs/remote/wsl). From there you can add extensions to VSCode, or open a terminal to access the WSL command line and install command line tools. 

## Debugging inside VSCode

We can debug our code from inside VSCode but it requires a little setup to make sure we're correctly using CMake when debugging. Follow [this tutorial to set up your VSCode properly with CMake](https://code.visualstudio.com/docs/cpp/CMake-linux).

## Debugging memory issues with Valgrind

If you're unlucky enough to have to resort to unsafe memory management with raw pointers, you will almost certainly meet a **segmentation fault** or segfault, if your program tries to access memory it doesn't strictly have access to. This can happen due to many different types of bugs; stack overflows, freeing already freed pointers, off-by-one bugs in loops, etc, but can be notoriously tricky to debug.

Valgrind is a **memory profiler and debugger** which can do many useful things involving memory but we just want to introduce its ability to find and diagnose segfaults by tracking memory allocations, deallocations and accesses.

You should follow [Valgrind's Quickstart Guide](https://valgrind.org/docs/manual/quick-start.html).

## Linting with clang-tidy

**Linters** are tools that statically analyse code to find common bugs or unsafe practices. We'll be playing with the linter from the Clang toolset, `clang-tidy` so follow this tutorial on setting up clang-tidy with VSCode:

{% include youtube_embed.html id="8RSxQ8sluG0" %}  

## Formatting with clang-format

If you've done much Python programming you probably already know the power of good formatters, tools that reformat your code to a specification. This can help standardise code style across codebases and avoid horrid debates about spaces vs tabs, where curly braces should go, or how many new lines should separate functions.

Again, we'll be peeking into the Clang toolbox and using `clang-format` to automatically format our code. Follow [this guide on setting up a basic .clang-format file](https://leimao.github.io/blog/Clang-Format-Quick-Tutorial/) and see clang-format's [list of common style guides](https://clang.llvm.org/docs/ClangFormatStyleOptions.html#basedonstyle) for more information about what styles are available. Look at a few, choose one you like and use that style to format your assignment code.

You can also use [the clang-format VSCode extension](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format) to automatically format your code on saving.

## Compiler warnings

One of the easiest ways to improve your code is to turn on **compiler warnings** and fix each warning. Some companies even require that all compiler warnings are fixed before allowing code to be put into production. Check out [this blog post on Managing Compiler Warnings with CMake](https://www.foonathan.net/2018/10/cmake-warnings/) for details on how to do this in our CMake projects. I recommend you use these warnings to fix potential bugs in your assignment.

## Optional: Performance profiling with gprof

Although you won't be required to use one on this course, as we move towards *performant* C++, one useful tool is a **profiler**. This is a tool that runs your code and measures the time taken in each function. This can be a powerful way to understand which parts of your code need optimising. 

There are many advanced profilers out there but a good, simple profiler is `gprof`. This also has the advantage of coming with most Linux distributions, so is automatically available with Ubuntu on either a native Linux machine or WSL. 

You can watch this introductory video on using gprof:

{% include youtube_embed.html id="zbTtVW64R_I" %}  

and try profiling one of your own codes. Since we're using cmake, we can't directly add the required `-pg` flags to the compiler so we'll have to tell cmake to add those flags with:

```
cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg ...
```

On MacOS you can try using Google's [gperftools](https://github.com/gperftools/gperftools) which is available through homebrew.

- You should target the areas of your code where your application spends the most time for optimisation. 
- Profilers are excellent for identifying general behaviour and bottlenecks, but you may be able to get more accurate results for specific functions or code fragments by inserting timing code. 