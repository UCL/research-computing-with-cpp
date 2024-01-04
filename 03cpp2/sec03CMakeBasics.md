---
title: CMake Basics
---

# CMake Basics

## What is CMake?

`CMake` is a _build system_, which can be used to build complex C++ projects in a more manageable way than manually compiling all the components using the command line. Once the build system is properly set up, recompilation just involved a single command. CMake can also keep track of which files have been changed and therefore which components need to be re-compiled and which don't. 

For most of the exercises and assignments we will provide the basic `CMake` files required, but you may need to make modifications to adapt to your programming and project organisation choices. 

## A Simple CMake File

We tell `CMake` what to do by using a file called `CMakeLists.txt`. A project with a more complex directory structure will involve multiple `CMakeLists.txt` files in different folders, but let's start with the assumption that all our files are in one folder.

In the same folder with your sources, you could have a file called `CMakeLists.txt` which looks as follows:

```
cmake_minimum_required(VERSION 3.21)
project(my_project_name
  VERSION 0.0.1
  LANGUAGES CXX
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(project_executable source1.cpp source2.cpp)
```

Let's break this down:

- `cmake_minimum_required` sets the minimum version of CMake that you expect a user to have. (You are unlikely to need to change this.)
- `project(...)` sets some project properties: the name, version, and the languages used. You will mostly be concerned with giving projects an appropriate name for now, although in the future _versioning_ is important to keep track of changes to an ongoing project. 
- `set(...)` sets a variable (first parameter) to a value (second parameter):
    - `CMAKE_RUNTIME_OUTPUT_DIRECTORY` is the folder to which CMake will output your executable. `${CMAKE_BINARY_DIR}` is a variable that refers to the root of the _build directory_, which is a parameter passed to `CMake` when we invoke the build command. This command therefore tells `CMake` to make a folder called `bin` inside the build folder, and put the executable there. 
    - `CMAKE_CXX_STANDARD` is the C++ standard used by this project. We will use C++17 for all projects in this course.
    - `CMAKE_CXX_STANDARD_REQUIRED` is a flag for forcing the compiler to use this standard; if this standard is not available then the compilation will fail. This is `OFF` by default. 
- `add_executable(...)` defines an executable (a program which can be run) which CMake should output. The first parameter is the name of the executable, and then it can take an arbitrary number of sources as subsequent parameters (separated by spaces with no commas). This is just the **source files** i.e. `.cpp` files, not the header (`.hpp`) files. 

If you have all the source and header files that you need in that top level folder, then you can easily build and run this program! Just type the following three commands in the terminal when in the directory with your sources:

```
mkdir build
cmake -B build
cmake --build build
```

These commands:
1. make a build folder,
2. configure CMake project, setting the build folder as the destination for any CMake outputs,
3. runs the build command, selecting the folder called `build` as the root build folder (see the note above about `CMAKE_BINARY_DIR`). This compiles your code and produces the executable. 

After doing this, you will have an executable that you can run in the folder `build/bin`. 

## Adding Folder Structure

A sensible way to break up a C++ project is to, at minimum, have a **source** folder and an **include** folder for your C++ files. 
- `.cpp` files go in `source`
- `.hpp` files go in `include`

You will usually also have a **test** folder for holding any test files. 
- These are also usually `.cpp` files, but these should only contain tests and should not be necessary for the normal function of your executable or library code. 

To use CMake with this kind of structure it's generally necessary to have additional `CMakeLists.txt` files. You _can_ do it all in one top level folder, but this file will start to get complicated and messy if you do! `CMake` allows us to handle things in a _modular_ way by breaking it up into multiple files. You can add a `CMakeLists.txt` file to both `source` and `test` (if you have a test folder). 

The top level `CMakeLists.txt` will be modified like this:
```
cmake_minimum_required(VERSION 3.21)
project(my_project_name
  VERSION 0.0.1
  LANGUAGES CXX
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(source)
add_subdirectory(test)
```
- `add_subdirectory` tells `CMake` that this project will also need to look into this other folder for additional `CMakeLists.txt` files. 
- Note that our executable is no longer defined in the top level! It usually makes more sense to define this in the `CMakeLists.txt` which contains the relevant source. 

In the `source` folder we then need a `CMakeLists.txt` file, and it should declare the executable:

```
add_executable(project_executable source1.cpp source2.cpp)
target_include_directories(project_executable ${CMAKE_SOURCE_DIR}/include)
```

- `add_executable` is as before.
- `target_include_directories` has been added so that CMake knows where to find the header files that it needs, now that they are not all in the same folder. `${CMAKE_SOURCE_DIR}` doesn't refer to the folder we have called `source`, but rather to the root folder from which `CMake` is invoked (the top level folder). So this command tells `CMake` to look for a folder called `include` inside the root folder, and find the headers in there. 

**We will look at how to install a testing framework, write tests, and use compile tests with CMake in the next section of the notes.**

With this basic structure in place you can already start making much more complex projects quite easily!

Sometimes you will have projects which need more folders than this, such as breaking a project up into libraries, which we will talk more about in week 6. 

- **N.B. I recommend having boiler-plate `CMakeLists.txt` files saved on your machine so you can set up projects quickly without having to memorise these commands.**
