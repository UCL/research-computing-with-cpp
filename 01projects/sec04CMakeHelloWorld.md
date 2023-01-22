---
title: HelloWorld with CMake
---

Estimated reading time: 40 minutes

# Building a simple C++ project with CMake

## Introduction

Having grasped the background and motivation for adopting the CMake build system generator, let us get our hands dirty and do a hands-on exercise in building a simple C++ project with CMake.
This will be CMake's equivalent of the classic hello world project, wherein we build an executable from C++ source code that prints ... well, "Hello World!" (what else?) to the screen.

We will perform this as the very first exercise during Week 1 of the class.

## Project Structure

To help focus on the new CMake-related aspects, we shall use a somewhat simplified version of the hierarchical project structure presented in the [previous lesson](./sec03CMakeBackground.md).

For this exercise, the following project structure is given:

```
./CMakeHelloWorld
├── LICENSE.txt
├── README.md
└── src
    └── hello.cpp

```

The `hello.cpp` file inside the `src/` folder contains a simple C++ source code that prints "Hello World!" to the console.

``` cpp

#include <iostream>

int main() {
    std::cout << "Hello World!" << '\n';

    return 0;
}

```

CMake processes user-provided configuration files that are named `CMakeLists.txt` (known as listfiles) in which we describe our build requirements at a high level.
Our task is to create and populate one or more `CMakeLists.txt` files that reflects our project's modular/hierarchical organisation.
The modern best practice is to always have a top-level `CMakeLists.txt` file at the project root, with additional, separate nested/hierarchical `CMakeLists.txt` files for each individual unit of logic/functionality i.e. resembling our project's file organisation.

Therefore, in this case, we shall use the following structure:

```

CMakeHelloWorld/
├── CMakeLists.txt
├── LICENSE.txt
├── README.md
└── src
    ├── CMakeLists.txt
    └── hello.cpp

```

### Populating the top-level (root) `CMakeLists.txt`

#### `cmake_minimum_required()`

`CMakeLists.txt` files are plain-text files written in the CMake-specific custom language syntax.
For every project, the top most `CMakeLists.txt` must start by specifying a minimum CMake version using the [`cmake_minimum_required()`](https://cmake.org/cmake/help/latest/command/cmake_minimum_required.html) command.
Although upper, lower and mixed case commands are supported by CMake, the modern trend is to use lower case commands.
Yet another thing to note is that, like C++ itself, `CMake` is whitespace-agnostic while processing `CMakeLists.txt` files.

Therefore, we start off with the following:
``` cmake
cmake_minimum_required(VERSION 3.25)
```

Even the most recent versions of CMake installations can replicate the exact behaviour of older releases.
In this example, the minimum `VERSION` is specified, and other options such as the maximum version the project is tested with have been omitted.

Owing to its long development history and strong emphasis on retaining backward compatibility, most CMake commands have multiple variants and sometimes have an overwhelming array of options.
Naturally, it is not feasible to cover every aspect of the language here.
However, we shall skim just enough of the language here to address our needs, and developers typically acquire proficiency in the language gradually over the years.
Therefore, we encourage you not to get disheartened by the slightly verbose documentation at the outset.

The main caveat regarding CMake's version used here is that the chosen minimum version should at least support C++17 language standards and compiler options.
Recent versions of CMake have improved package detection, simplified dependency handling.
While an older version suffices for this simple HelloWorld example, it does not hurt to use a CMake version as recent as possible.

#### `project()`

Next, we use the `project()` command to set the project name.
This is required with every project and should be called soon after `cmake_minimum_required()`.
This command can also be used to specify other project level information such as the language or version number.

``` cmake
project(hello_world
  VERSION 0.0.1
  LANGUAGES CXX
)
```

The first (required) argument to this command is the project's name.
The name of the project can be any valid string.
It does not have to be the same string as the name of the main executable program to be built (although it is conventional/convenient to use the same string for them).
The VERSION field is optional, but is conventional to populate it.
Next, we specify the programming languages that the project uses.
The keyword for C++ projects is `CXX`.
It is helpful to specify the list of languages used in the project, so that cmake does not spend time inspecting the system for the presence of the toolchains for every supported language (CUDA, fortran, C, Java etc).

#### Setting global options at the project root with `set()`

It was common practice to make liberal use of the [`set()`](https://cmake.org/cmake/help/latest/command/set.html) command to define various built-in, user-defined and environment variables at the top level `CMakeLists.txt`.
This includes setting up the C++ language standard, compiler flags etc.
While it can be useful to do so, CMake provides a much more fine-grained control to set options for each logical/modular unit in our project, and we have a slight preference towards this approach.

Nevertheless, the top level `CMakeLists.txt` file can be used to set various global conditions applicable to the whole project.
When CMake configures a project, the build tree closely resembles the source tree hierarchy in the build tree.
The path to the root of the build directory is available as the built-in variable [`CMAKE_BINARY_DIR`](https://cmake.org/cmake/help/latest/variable/CMAKE_BINARY_DIR.html) whose value can be accessed within `CMakeLists.txt` by dereferencing it using the syntax `${VAR_NAME}`, in this case `${CMAKE_BINARY_DIR}`.
However, a common practice for implementing a logical separation is to have CMake place the generated executables within a separate `bin/` subfolder at the root of the build directory.
This can be achieved by setting the value of the [`CMAKE_RUNTIME_OUTPUT_DIRECTORY`](https://cmake.org/cmake/help/latest/variable/CMAKE_RUNTIME_OUTPUT_DIRECTORY.html) as follows

``` cmake
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
```

#### Parsing nested `CMakeLists.txt` within the `src/` directory

Until now, we have set certain high-level description of the project, but have not provided instructions on building the "hello_world" executable program from the source!
This is intentional.
The task of describing the build of each logical/modular project unit is delegated to nested/leaf `CMakeLists.txt` that reflects the project's hierarchy.
In this case, we wish to 'hand-off' control to `src/CMakeLists.txt` for building the `hello_world` executable.
This is achieved by using the [`add_subdirectory()`](https://cmake.org/cmake/help/latest/command/add_subdirectory.html) command.

``` cmake
add_subdirectory(src)
```

In the above line, we instruct CMake to process the `CMakeLists.txt` located within the `src/` directory.
These list files can be nested quite deeply, and at each depth level, we can have as many 'sibling' folders with one `CMakeLists.txt` per folder that describes the logical program unit to be built.

After processing all nested list files, control returns to the top level `CMakeLists.txt`.
Since the above line was the last in this file, the configuration process is completed.
We now focus on the build details that is to be described in `src/CMakeLists.txt`.

### Contents of `src/CMakeLists.txt`

Within this 'leaf' `CMakeLists.txt` file, we describe how the `hello_world` executable is to be built.
This is considered best practice, i.e. to have a `CMakeLists.txt` as close to the relevant source files for each modular functionality within the project.
This file can see and use all the properties, variables, functions and macros defined at any `CMakeLists.txt` in its parent scope (in this case, all definitions from the top-level `CMakeLists.txt` file are visible here i.e. they are in the 'scope' of this leaf file).

#### Targets and usage requirements

We like CMake to build for us an executable called 'hello_world'.
In this case, `hello_world` is a CMake 'target' (or a CMake 'executable target' to be precise).
Targets (not variables) are the first-class citizens in modern CMake language.
Hence, the relationships between various constituents (i.e. executables, source files, header files, internal libraries, other third-party code/files, system dependencies etc.) is to be described in a target-centric way within each `CMakeLists.txt` file.

The behaviour of targets, their features and their various properties can be set in three ways:
- only on/for/affecting itself (using the PRIVATE keyword)
- only on/for/affecting other targets that consume this current target (using the INTERFACE keyword)
- on/for/affecting itself as well as other targets that consume this current target (using the PUBLIC keyword)

We recognise that this is quite a confusing aspect for newcomers to CMake, but is one of the key things to understand how targets and their dependencies interact for a given build.
The only way to get better comprehend this critical component is by practicing these ideas during the first few weeks of the course.

The following matrix may help slightly, and we may choose to refer to this at times throughout the course:

Keyword selection table (Who needs/uses a dependency?)

|                | Other (consuming targets) |
|----------------|------------|---------|
| **Current target** | **YES**        | **NO**      |
| **YES**            | PUBLIC     | PRIVATE |
| **NO**             | INTERFACE  | N/A     |

#### The `add_executable()` command

We use the [`add_executable()`](https://cmake.org/cmake/help/latest/command/add_executable.html) command to declare that we'd like CMake to build an executable program called 'hello_world' (executable target in CMake parlance).

``` cmake
add_executable(hello_world)
```

#### The `target_sources()` command

Next, we tell CMake that the executable target `hello_world` depends on one C++ source file, viz `hello.cpp`.
This is done via the [`target_sources()`](https://cmake.org/cmake/help/latest/command/target_sources.html) command.

``` cmake
target_sources(hello_world PRIVATE hello.cpp)

```

Referring to the usage requirements table, the keyword `PRIVATE` conveys to CMake that the file `hello.cpp` serves as a dependency only for the current target `hello_world`, and is not propagated to any other targets that depend on `hello_world`.
This is not much relevant here, since the `hello_world` target is simply an executable program which does not depend on the file `hello.cpp` after it has been built (i.e. at run-time).
In such circumstances, the keyword can be omitted.
However, the usage requirement keywords become crucial (and sometimes mandatory) in describing more intricate dependency graphs involving various libraries as projects evolve in complexity over time.
Hence, we strongly encourage students not to omit this keyword and think about the usage requirement for every target property/behaviour set in `CMakeLists.txt`.

#### Setting the compile time properties of targets

We wish to ensure that the `hello_world` target conforms to the C++17 standard.
One way to achieve this in CMake is through the [`target_compile_features()`](https://cmake.org/cmake/help/latest/command/target_compile_features.html) command which can be used to set some pre-defined properties/behaviour/characteristics to follow for compiling targets.

We add the following line:

``` cmake
target_compile_features(hello_world PUBLIC cxx_std_17)

```

Here the keyword `cxx_std_17` keyword gently suggests CMake to invoke a C++17 compatible compiler if detected on the target system.
One of PRIVATE or PUBLIC keyword is required here (the INTERFACE keyword does not directly apply to executable targets).
We may use either here, but we chose PUBLIC to be compatible with recommended best practices for new projects.
In the future, if we wish to split our project into a library the PUBLIC keyword will ensure that our library as well as any other executables or other libraries that depend on this shall acquire C++17 capabilities.

#### Setting individual properties on targets

While `target_compile_features()` allows us to select a list of high-level behaviour, an alternate is to set individual properties/flags on each target through some pre-defined CMake variables.
This is achieved using the [`set_target_properties()`](https://cmake.org/cmake/help/latest/command/set_target_properties.html) command.

While `target_compile_features(hello_world PUBLIC cxx_std_17)` encourages the use of C++17  standards, it does not enforce this.
The built-in boolean [`CXX_STANDARD_REQUIRED`](https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD_REQUIRED.html) property can be set to `True/ON` to enforce this.
Another useful thing to do to ensure wide platform and target portability of our project is to disable compiler extensions.
Various C++ compilers take liberties to enable compiler-specific extensions by default unless explicitly directed not to do so.
The built-in boolean property [`CXX_EXTENSIONS`](https://cmake.org/cmake/help/latest/prop_tgt/CXX_EXTENSIONS.html) can be set to `False/Off` to disable these compiler extensions.
The `set_target_properties()` command allows us to set multiple properties on targets within a single invocation like so:

``` cmake
set_target_properties(hello_world PROPERTIES CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

```

A starter template/scaffolding for the project has been provided with relevant prompts for filling in the required CMake commands in the two `CMakeLists.txt` files.
We shall perform this during the first class.
We have now fully described our build, and are now ready to proceed to perform the build for our project, all whilst conforming to the desired C++ standard and implementing CMake best practices!
The procedure to build and run the executable is described in the next lesson.
