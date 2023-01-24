---
title: CMake Background
---

Estimated reading time: 35 minutes

## Introduction to modern CMake

### CMake versions

Since its initial release, CMake has continuously evolved and improved.
The developers make regular feature updates to CMake to facilitate modern software development principles.
So, in practice what does version of CMake can be considered as 'modern'?
Version `2.18.12` (which served as the starting point of the `v3.x` series) that appeared nearly 15 years ago, introduced many paradigm shifts, but can no longer be considered modern by today's standards.
However, it is being mentioned here for the clean break it enforced from the 2.x series (much akin to the move to `python 3.x` from `python 2.x`).

Recall that CMake is a build system generator, and not a build system.
It queries the hardware and environment for the toolchain capabilities to use to generate a build system that invokes these tools with appropriate functionality (i.e. options) enabled.
It is important to **never** use a CMake version **older than the intended compiler/toolchain**.
This is because CMake will not understand the additional functionality that the compiler is capable of.
In this course, we shall be using the **C++17** standard, for our projects.
Nearly all major compiler vendors have implemented nearly complete support for this standard since ~2019.
For instance, the [`gcc`](https://en.wikipedia.org/wiki/GNU_Compiler_Collection) toolchain, ubiquitously available on all Unix-like systems has nearly complete support for C++17 since ~9.x.
Most C++17 features were implemented by the [LLVM/Clang toolchain](https://en.wikipedia.org/wiki/Clang#Status_history) by ~v9, and by Microsoft's C++ toolchain (available for Windows systems) since VS2017 15.x.
The toolchains provided by HPC system vendors such as Intel and Cray have also supported C++17 over the last 2 years on powerful compute clusters.

Initial C++17 support was added to CMake in version 3.8 (2017) and improved steadily in the next few releases.
This, coupled with improved compiler support for the C++17 standard means that, in principle, any CMake version released in the last 2 years or so shall be more than sufficient for new projects.
CMake facilitates full backwards compatibility i.e. a newer CMake version is able to exactly replicate the behaviour of older versions through a simple user configuration command (which we will cover soon).
Recent versions of CMake continue to gain support for GPUs (with better support for CUDA) and other modern technologies.
Furthermore, the most recent version of CMake is extremely easy to install (even without administrative privileges), and given its ability to easily enforce backward compatibility, there is no reason not to install the latest version (which as of Dec 2022 is `3.25.1`).

### Installing CMake

In this course, we shall use the latest available version of CMake.
The VSCode development container provided for the course already has this latest version installed.
However, since we do not invoke the very latest features in the initial weeks of the course, we shall configure it to use a slightly older version for broad compatibility with most platforms and shall help with installing the latest version.

CMake can be installed through a variety of methods on virtually every platform (processor architecture and OS).
Detailed install instructions shall be made available on the Moodle portal for the course.

## CMake overview

As a high level overview, CMake is a build system generator which provides command line tools that read user-provided plaintext files describing the build requirements of the project, and produces files that can be consumed by a native build system for the platform (e.g. Makefiles).
The plaintext files that describe the project's build requirements are written in CMake's custom language and are named `CMakeLists.txt` (the filenames must have exactly identical casing as given here).
It can be considered as yet another programming/scripting language with its own rubrics for variables, functions, macros and other control structures (conditional statements, loops etc).
Owing to the need to cater to a wide variety of platforms, the build requirements are written at a rather high-level describing *what* must be built, and describes the relationships between various dependencies in the projects, and excludes low-level minute details of *how* something should be built.

CMake is much more comprehensive than what was described here, and provides its own integrated support for not only generating the build system, but also performing the build with great control over various aspects, as well as facilitating testing, installation of software onto standard locations on the user's machine, packaging the build's artifacts for wider distribution on various platforms, and various other functionality.
We recognise the cognitive burden of learning yet another full-fledged language just to describe the project's build requirements!
Hence, we shall strive to introduce the CMake language and tooling piecemeal over the duration of this course.

### Hierarchical project structure

Large scale research projects typically use a hierarchical project structure using modular code patterns for [separation of concerns](https://en.wikipedia.org/wiki/Separation_of_concerns). A modular code pattern refers to a software design paradigm wherein each unit of code has well-defined interfaces and one (and only one) specific functionality. The modularity aspect arises from the fact that another code unit with a different/improved algorithm may be swapped in for an existing code unit, whilst retaining the same interfaces and system behaviour.
Here, we discuss an example hierarchical project structure, albeit at a smaller scale that you may adopt suitably, depending on the current needs.

```

./my_app/
├── CMakeLists.txt
├── LICENSE.txt
├── README.md
├── cmake/
├── external/
├── src/
└── test/

```

#### Components that do not directly influence the local build

The project's root primarily contains meta-information (e.g. for human consumption).
For instance, professional projects have a `LICENSE.txt` file that describes the wording of the license that covers the terms and conditions of usage of the code and (if any) derivatives.
The `README.md` file provides a high-level description of the project's aims and purposes, and provides instructions for its usage (e.g. how to build the project, provide user inputs, configuration options, running etc).
For detailed documentation, some scientific projects have a `doc(s)/` folder with the plaintext sources to generated documentation (e.g. rendered as HTML or converted to PDF).
We do not explore this detailed documentation aspect in this course.
Other typical files placed in the project's root are a contribution guide (for open source projects), describing the procedures and practices for accepting code into the project for bug fixes and enhancements.

Professional projects employ a Version Control System (VCS) such as `git`, for meticulously tracking the changes to project-related files.
The tracked components of the project tree along with components to handle its version history on the local filesystem constitutes the local project *repository*.
The root of the project tree also contains VCS-specific files and folders such as `.git/` and `.gitignore`.
The project root might have configuration files pertaining to tools for automating tasks such as running tests on a remote machine, performing static analysis, ensuring consistent formatting of source code etc.
We shall cover these topics and introduce associated tools in future lessons.

#### Components that directly influence the local build

At the root of the project tree, we have a top-level `CMakeLists.txt` file that describes the high-level build details of the overall project.
We describe the basic contents and organisation of this file in the next lesson.
To keep the length of the top-level `CMakeLists.txt` file manageable and for readability purposes, we may optionally choose to split certain boiler-plate project-level CMake helper scripts, functions and other custom CMake-specific files in the `cmake/` directory.

The `external/` folder is intended for code and data from external sources e.g. a third-party library, that our project depends on, and we provide appropriate instructions to CMake on how to incorporate that in building our project.
Note, in particular, that there are no source files at the root of the project repository.
The build process itself could produce/create other (ephemeral or otherwise) generated files and folders.
Such build artefacts are not considered as part of the core project files, and are configured to be ignored in version control systems.
All project-specific source files are located inside the `src/` directory and codes for testing the programs' logic are placed in the `test` directory.
We shall present the details of the `src/` tree next, and defer discussing the `test/` tree until later (when we introduce unit testing).

##### The `src/` tree

An exploded view of a typical source tree can be as follows:

```

./my_app/src/
├── CMakeLists.txt
├── include/
│   ├── common_libheader.h
│   └── my_app.h
├── lib1/
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── header_lib1.h
│   └── src_lib1.cpp
├── lib2/
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── header_lib2.h
│   └── src_lib2.h
└── my_app.cpp

```

In this `src/` project structure, the `my_app.cpp` contains the `main()` entry point function of our C++ project.
We assume that we are modularising our overall code by grouping related functionality into two libraries `lib1` and `lib2` that provide non-overlapping/independent features that can be reused elsewhere within the project (e.g. in `my_app.cpp`).
The library-specific header files live within the nested `include/` directories of each of their respective folders, while common header files that both libraries and the main executable may use are kept in `src/include`.
The `CMakeLists.txt` files located within each library's directory describes the building of that library alone.
The `CMakeLists.txt` within `src/` describes how to build `my_app` from the corresponding sources, and how to link up the two internal libraries and any external third party libraries (from `../external`) to produce the final executable.

Such a modular project structure, and the correspondingly nested `CMakeLists.txt` that reflects this hierarchy, help in clean separation of functionality into distinct pieces, and aids in the comprehension of dependencies in large projects, as well as facilitating the testing and debugging of such small well-defined project modules.
While many variations of the aforesaid project structure exist, it is generally acknowledged to use  such a hierarchical/modular project structure which results in simpler build descriptions using CMake. In the next lesson, we start writing our own `CMakeLists.txt` for a simple project.
