---
title: CMakeCatchTemplate
---

## CMakeCatchTemplate

### Intro 

* Demo project on [GitHub](https://github.com): [CMakeCatchTemplate](https://github.com/MattClarkson/CMakeCatchTemplate)
* No functional code, other than adding 2 numbers
* Basically shows how to use CMake


### Features

* Full feature list in [README.md](https://github.com/MattClarkson/CMakeCatchTemplate/blob/master/README.md)
* SuperBuild or use without
* Downloads Boost, Eigen, glog, gflags, OpenCV, PCL, FLANN, VTK
* Example GUI apps (beyond scope of course)
* Unit testing
* Script ```rename.sh``` to create your own project


### Main CMake flags

These were covered earlier - here's how you set them.

* ```CMAKE_BUILD_TYPE:String=[Debug|Release]```
* ```BUILD_SHARED_LIBS:BOOL=[OFF|ON]```
* Compile flags
    * ```CMAKE_C_FLAGS, CMAKE_CXX_FLAGS, CMAKE_EXE_LINKER_FLAGS``` all ```String```
    
### 32 or 64 bit

* Generally stick with default
* 32 or 64 bit?
    * Visual Studio - chose generator, check in Visual Studio (x86, Winn64)
    * ```CMAKE_C_FLAGS = -m32``` etc. [For example](https://unix.stackexchange.com/questions/352783/how-can-i-build-and-run-32-bit-software-on-64-bit-debian)


### SuperBuild

* See flag: ```BUILD_SUPERBUILD:BOOL=[ON|OFF]```
* If ```OFF```
    * Just compiles *this* project in current folder
* If ```ON```
    * Dependencies in current folder
    * Compiles *this* project in sub-folder
* Try it
* A SuperBuild with no dependencies just does nothing


### Home Work 1a - Basic Build 

* Build CMakeCatchTemplate ```BUILD_SUPERBUILD=OFF```
* Modify some C++, recompile
* Choose a project relevant to you - eg. ITK
* Compile it separately
* Or install via package manager


### Home Work 1b - Basic Build

* Try to ```find_package(PackageName)``` to see what happens
* Set ```include_directories```
* Add library to ```target_link_libraries```
* Work out if its necessary to build it yourself


### Home Work 2 - SuperBuild

* Not for the faint-hearted
    * 3rd party examples in ```CMake/ExternalProjects```
    * Create a new one for your project
    * Add your project name to loop in ```SuperBuild.cmake```
    * Add top-level control variables like ```BUILD_MyExternalProject```
    * Pass variables from SuperBuild to main project build, as shown in ```SuperBuild.cmake```
