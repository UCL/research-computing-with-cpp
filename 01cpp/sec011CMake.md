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


### CMake Usage Linux/Mac

Demo an "out-of-source" build

```
cd ~/build
git clone https://github.com/MattClarkson/CMakeHelloWorld
mkdir CMakeHelloWorld-build
cd CMakeHelloWorld-build
ccmake ../CMakeHelloWorld
make
```


### CMake Usage Windows

Demo an "out-of-source" build

* git clone https://github.com/MattClarkson/CMakeHelloWorld
* Run cmake-gui.exe
* Select source folder (CMakeHelloWorld downloaded above)
* Specify new build folder (CMakeHelloWorld-build next to, but not inside CMakeHelloWorld)
* Hit *configure*
* When asked, specify compiler
* Set flags and repeatedly *configure*
* When *generate* option is present, hit *generate*
* Compile, normally using Visual Studio

