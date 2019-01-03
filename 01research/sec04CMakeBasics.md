---
title: CMake Basics
---

## CMake Basics

### Compiling Basics

Question: How does a compiler work?


### How does a Compiler Work?

Question: How does a compiler work?

* (Don't quote any of this in a compiler theory course!)
* .cpp/.cxx files compiled, one by one into .o/.obj
* executable compiled to .o/.obj
* executable linked against all .o, and all libraries 

That's what you are trying to describe in CMake.


### CMake - Directory Structure

* CMake starts with top-level CMakeLists.txt
* CMakeLists.txt is read top-to-bottom
* All CMake code goes in CMakeLists.txt or files included from a CMakeLists.txt
* You can sub-divide your code into separate folders.
* If you ```add_subdirectory```, CMake will go into that directory and start
to process the CMakeLists.txt therein. Once finished it will exit, go back
to directory above and continue where it left off.
* e.g. top level CMakeLists.txt
```
project(MYPROJECT VERSION 0.0.0)
add_subdirectory(Code)
if(BUILD_TESTING)
  set()
  include_directories()
  add_subdirectory(Testing)
endif()

```

### CMake - Define Targets

* Describe a target, e.g. Library, Application, Plugin
```
add_executable(hello hello.cpp)
```
* Note: You don't write compile commands
* You tell CMake what things need compiling to build
a given target. CMake works out the compile commands!


### CMake - Order Dependent

* You can't say "build Y and link to X" if X not defined
* So, imagine in a larger project
```
add_library(libA a.cpp b.cpp c.cpp)
add_library(libZ x.cpp y.cpp z.cpp)
target_link_libraries(libZ libA)
add_executable(myAlgorithm algo.cpp) # contains main()
target_link_libraries(myAlgorithm libA libZ ${THIRD_PARTY_LIBS})
```
* So, logically, its a big, ordered set of build commands.


### Classroom Exercise 3. (or Homework)

* Build https://github.com/MattClarkson/CMakeLibraryAndApp.git
* Look through .cpp/.h code. Ask questions if you don't understand it.
* What is an "include guard"?
* What is a namespace?
* Look at .travis.yml and appveyor.yml - cross platform testing, free for open-source
* Look at myApp.cpp, does it make sense?
* Look at CMakeLists.txt, does it make sense?
* Look for examples on the web, e.g. [VTK](https://lorensen.github.io/VTKExamples/site/Cxx/GeometricObjects/Cone/)
