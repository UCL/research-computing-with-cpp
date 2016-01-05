---
title: CMake
---

## CMake

### CMake Introduction

* This is a practical course
* We will use CMake as a build tool
* CMake produces
    * Windows: Visual Studio project files
    * Linux: Make files
    * Mac: XCode projects, Make files
* This course will provide CMake code and boiler plate code


### CMake Usage 1

Typically, to do an "out-of-source" build

```
cd ~/myprojects
git clone http://github.com/somecode
mkdir somecode-build
cd somecode-build
ccmake ../somecode
make
```

### CMake Usage 2

* Set flags and repeatedly cmake
* Then once set, hit compile
* Best to demo this

