---
title: "Week 6: Tooling"
---

## Unit testing

Testing single functions, methods or classes is referred to as **unit testing**, i.e. testing single *units* of code. We've already seen some unit tests earlier in this course written with the unit testing framework [Catch2](https://github.com/catchorg/Catch2) so you should have some understanding how to write, compile and run unit tests. However, like many things in programming, there is an art to writing *good* unit tests that provide enough **test coverage**, that is the amount of code tested by unit tests. To dig deeper into the philosophy of testing C++ code, we recommend the following talk from the 2022 Cppcon:

{% include youtube_embed.html id="SAM4rWaIvUQ" %}  

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

## Performance profiling with gprof

As we move towards writing *performant* C++, one essential tool is a **profiler**, a tool that runs your code and measures the time taken in each function. This can be a powerful way to understand which pieces of your code need optimising.

There are many advanced profilers out there but a good, simple profiler is `gprof`. Watch this introductory video on using gprof:

{% include youtube_embed.html id="zbTtVW64R_I" %}  

Now try profiling one of your own codes. Since we're using cmake, we can't directly add the required `-pg` flags to the compiler so we'll have to tell cmake to add those flags with:

```
cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg ...
```
