---
title: CMake
---

## CMake

### Ever worked in Industry?

* (0-3yrs) Junior Developer - given environment, team support
* (4-6yrs) Senior Developer - given environment, leading team
* (7+ years) Architect - chose tools, environment, design code
* Only cross-platform if product/business demands it
* All developers told to use the given platform no choice


### Ever worked in Research?

* All prototyping, no scope
* Start from scratch, little support
* No end product, no nice examples
* Cutting edge use of maths/science/technology
* Share with others on other platforms
* Develop on Windows, run on cluster (Linux)


### Research Software Engineering Dilemma

* Comparing Research with Industry, in Research you have:
    * Least experienced developers
    * with the least support
    * developing cross-platform
    * No clear specification or scope
    
* Struggle of C++ is often not the language its the environment


### Build Environment

* Windows: Visual Studio solution files
* Linux: Makefiles
* Mac: XCode projects / Makefiles

Question: How was your last project built?

    
### CMake Introduction

* This is a practical course
* We will use CMake as a build tool
* CMake produces
    * Windows: Visual Studio project files
    * Linux: Make files
    * Mac: XCode projects, Make files
* So you write 1 build language (CMake) and run on 
multi-platform.
* This course will provide most CMake code and boiler plate code
for you, so we can focus more on C++. But you are expected
to google CMake issues and work with CMake.


### CMake Usage Linux/Mac

Demo an "out-of-source" build

``` bash
mkdir cmake_demo
cd cmake_demo
git clone https://github.com/MattClarkson/CMakeHelloWorld
cd CMakeHelloWorld
mkdir build
cd build
cmake ..
make
```

### Homework - 2

* Build https://github.com/MattClarkson/CMakeHelloWorld.git
* Ensure you do "out-of-source" builds
* Use CMake to configure separate Debug and Release versions
* Add code to hello.cpp:
    * On Linux/Mac re-compile just using make

### Homework - 3

* Build https://github.com/MattClarkson/CMakeHelloWorld.git
* Exit all code editors
* Rename hello.cpp
* Change CMakeLists.txt accordingly
* Notice: The executable name and .cpp file name can be different
* In your build folder, just try rebuilding.
* You should see that CMake is re-triggered, so you get a cmake/compile cycle.



