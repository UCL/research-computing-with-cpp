---
title: CMakeCatch2
---

## CMakeCatch2

### Intro 

* Demo project on [GitHub](https://github.com): [CMakeCatch2](https://github.com/MattClarkson/CMakeCatch2)
* No functional code, other than adding 2 numbers
* Sets up Eigen and Catch, header only, checked into project


### Homework - 9a

* Download ```CMakeCatch2```
* Practice compiling ```CMakeCatch2``` project, testing the flags above.
* Choose a another project relevant to you - eg. glog/gflags/VTK/OpenCV
* Compile it in a separate folder, as per their build instructions
* Modify some C++ in ```CMakeCatch2```, recompile
* Try to include external library, e.g. glog/gflags/VTK/OpenCV and call a simple function
    * See example for Eigen, in [Command Line App](https://github.com/MattClarkson/CMakeCatch2/blob/master/Code/CommandLineApps/mpMyFirstApp.cpp)


### Homework - 9b

These were covered earlier - here's how you set them.

* ```CMAKE_BUILD_TYPE:String=[Debug|Release]```, sets different optimisation levels, and debug symbols
* ```BUILD_SHARED_LIBS:BOOL=[OFF|ON]```, switch between static linking and dynamic linking
* Compile flags
    * ```CMAKE_C_FLAGS, CMAKE_CXX_FLAGS, CMAKE_EXE_LINKER_FLAGS``` all ```String```
    * You can add extra compile flags as you wish.
    

### Homework - 9c

* 32 or 64 bit?
* Generally stick with default for your platform
* If you need to change:
    * Visual Studio - chose generator, check in Visual Studio (x86, Winn64)
    * Linux ```CMAKE_C_FLAGS = -m32``` etc. [For example](https://unix.stackexchange.com/questions/352783/how-can-i-build-and-run-32-bit-software-on-64-bit-debian)

