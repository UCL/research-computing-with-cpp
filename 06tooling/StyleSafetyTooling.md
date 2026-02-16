---
title: Tooling
---

# Tooling 

This week we will introduce a number of tools which are useful when developing and analysing C++ code. None of them are strictly necessary, but each serves a useful purpose and helps to detect or fix problems that can be difficult to find or keep on top of otherwise. 

If you are using the **Docker container** then you will not need to install any of these tools manually; the container will already have pre-installed valgrind, clang-tidy, and clang-format. However, if you do not wish to use Docker you can install these tools yourself. 

## Installation (only if not using Docker)

**N.B.** Please remember that if you are using Windows for this course you will need to install these tools **inside WSL** (Windows Subsystem for Linux) rather than following a normal Windows installation. To do so, you can 
1. Open a Command Prompt and type `wsl` to go into the WSL command line. From there you can follow Linux instructions for installing command line tools like Valgrind. 
2. Open VSCode and [connect to WSL using the button in the bottom left hand corner](https://code.visualstudio.com/docs/remote/wsl). From there you can add extensions to VSCode, or open a terminal to access the WSL command line and install command line tools. 

### WSL / Linux

On WSL/Linux you can install all three using the following commands in the terminal:

```
apt-get -y install --no-install-recommends valgrind clang-tidy clang-format 
```

You may need to add `sudo` before these commands depending on the permissions on your system. If it does not find any of these tools you may need to update your packaged index using:

```
apt-get update
```

### MacOS

If you want to use MacOS without using the devcontainer then the process is a little more involved. 

- Valgrind may not be compatible with your device depending on your OS and architecture. XCode command line tools however comes with a similar tool called `leaks` which you can use instead to detect memory leaks.
- Clang-format can be installed by using homebrew (`brew install clang-format`), but clang-tidy cannot. You can install both `clang-format` and `clang-tidy` as part of llvm (`brew install llvm`), but llvm is a large install, and you will have to then add these to your path. 

## Compiler warnings

Even without any further tooling, your compiler can already provide a lot of really useful code analysis that can improve the quality of your code and catch various mistakes. For a detailed look at your compiler warning options you can take a look at the [g++ warning options documentation](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html). 

Compiler warnings can be turned on for `g++` by using some of the following flags:

- `-Wall`: enables common warnings for things like functions with missing return statements, uninitialised variables, unused variables. Some less common (but extremely useful) warnings in this category include warnings for strict aliasing, dangling pointers/references, and array bounds (for arrays with static allocation), although these kinds of memory related checks are imperfect and cannot catch all cases of bad behaviour. 
- `-Wextra`: enables some additional warnings that are less common such as unused function parameters. For some of these warnings to function correctly you will also need `-Wall` turned on.
- `-Wconversion`: enables warnings for implicit conversions that can alter values such as `double` to `float`, but not for sign conversions.
- `-Wsign-conversion`: enables warnings for implicit conversions between signed and unsigned types that may cause underflow/overflow errors.
- `-Werror`: turns all warnings into compilation errors so that the program will not compile until all the warnings are resolved.

I would recommend using all of these flags by default, especially when checking things such as your coursework, although it is sometimes useful to turn off `-Werror` during development (for example, when you have a variable that is unused because you haven't got to the stage in an algorithm where it is going to be used, but you want to compile in the meantime).  

## Debugging inside VSCode

You can use the debugger inside VSCode, or you can just `gdb` in the command line to debug your programs. We covered how to set up your debugger in the first week's class, so you can go back to that if you need a refresher. Microsoft also provides [this tutorial for setting up VSCode with cmake](https://code.visualstudio.com/docs/cpp/CMake-linux), which is a little more work to set up but can make it easier to integrate building and debugging into the VSCode environment. 

## Debugging memory issues with Valgrind (or `leaks`)

We've discussed in the first half of this course many of the problems that can arise from improperly managed memory, including memory leaks and segmentation faults (which occur due to invalid memory accesses). Valgrind is a **memory profiler and debugger** which is extremely useful for analysing our programs memory usage by keeping track of allocations and deallocations (amongst other things). This can be vital for checking that we don't have memory leaks when we run our program. Memory leaks can be especially hard to spot, since they typically grow over time during a program's execution and only cause the program to abort if you run out of memory. This means that short programs such as tests rarely use up enough memory to make a leak obvious, only for the problem to cause a job on a computer cluster to fail after days of running because the memory has been exhausted. By detecting when memory is allocated and deallocated, valgrind can check if we have failed to free any memory _even in short test cases_. 

A limitation of valgrind is that it does _not_ guarantee that your program is free of problems; it only reports what happened _this time_ when you ran the program. If you have a memory leak that only happens under certain circumstances, it won't catch it unless it occurs during a run that you are analysing with valgrind. Nevertheless it is extremely useful, especially as 

Valgrind provides a [quick-start guide](https://valgrind.org/docs/manual/quick-start.html) which demonstrates how to run valgrind with a basic C++ program. It demonstrates the detection of a _heap block overrun_ (i.e. if we have written into invalid memory by going beyond the end of an array) and a _memory leak_ (a heap allocation that has not been freed).

If you are using MacOS you can use `leaks` in a similar way. 

## Linting with clang-tidy

**Linters** are tools that statically analyse code to find common bugs or unsafe practices; it is significantly more powerful than just using the g++ compiler warnings. Microsoft has provided a tutorial for setting up the `clang-tidy` tool with VSCode:

{% include youtube_embed.html id="8RSxQ8sluG0" %}  

You can also just run clang-tidy from the command line:

```
clang-tidy main.cpp -- -std=c++17
```

will provide the default analysis of a code file. You can see what checks clang-tidy does by default using:

```
clang-tidy -list-checks
```

## Formatting with VSCode or clang-format

Formatters make sure that your program is presented with consistent style with respect to things like line length / line breaks, indentation, and spacing. These are vital for making sure that your code is easy to read and understand, and also makes sure that teams of programmers working together can construct a code base that is consistent in style. This has the added benefit in collaboration of making sure that changes to the code that need to be reviewed are not clogged by meaningless style changes when two people who can't agree on a coding style are working on the same pieces of code! 

In VSCode, if you have the C++ extension pack installed, you should be able to format your code using:

- Highlight the code that you want to format or use ctrl-A to highlight all, then ctrl-K followed by ctrl-F to format it.

You can also use a tool called `clang-format` to format code from the terminal, and there is a `clang-format` VSCode extension as well. There is a [clang-format guide](https://clang.llvm.org/docs/ClangFormat.html) for getting started using it if you want to integrate it into projects. 

The simplest start is to just run `clang-format` from  the command line. If you run it with no flags, it will simply write the result to the terminal. You can use the `-i` flag (for "in place") to have clang-format modify your source files directly. For example:

```
clang-format -i main.cpp
```

Formatters are typically configurable, and `clang-format` provides a number of different presets based on [common style guides](https://clang.llvm.org/docs/ClangFormatStyleOptions.html#basedonstyle); for our purposes it's not very important which one of these that you use provided that you are consistent and your code is clearly presented. 




