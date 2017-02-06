---
title: Linking Libraries
---

## Linking libraries

### Aim

* Can be difficult to include a C++ library
* Step through
    * Ways of setting paths/library names
        * Platform specific 
        * Use of CMake
* Aim for - source code based distribution


### Recap

* So far
    * Seen how to chose
    * Static / Dynamic
    * How to structure layout
* Now we want to use a library


### Fundamental

* Access to header files - declaration
* Access to compiled code - definition


### Linux/Mac

```
g++ -c -I/users/me/myproject/include main.cpp
g++ -o main main.o -L/users/me/myproject/lib -lmylib
```

* ```-I``` to specify include folder
* ```-L``` to specify library folder
* ```-l``` to specify the actual library

Notice: you don't specify .a or .so/.dylib if only 1 type exists.


### Windows

* Visual Studio (check version)
* Project Properties
    * C/C++ -> Additional Include Directories.
    * Configuration Properties -> Linker -> Additional Library Directories
    * Linker -> Input -> Additional Dependencies.
* Check compile line - its equivalent to Linux/Mac, -I, -L, -l


### Header Only?

* After all this:
    * Static/Dynamic
    * Package Managers / Build your own
    * External build / Internal Build
    * Release / Debug
    * -I, -L, -l
* Header only libraries are very attractive.


### Use of CMake

* Several ways depending on your setup
    * Hard code paths and library names
    * Use a FindModule
    * Generate and use a FindModule


### CMake - Header Only

* catch.hpp Header files in project [CMakeCatchTemplate](https://github.com/MattClarkson/CMakeCatchTemplate/blob/master/Testing/mpBasicTest.cpp)
* From ```CMakeCatchTemplate/CMakeLists.txt```

```
include_directories(${CMAKE_SOURCE_DIR}/Code/)
add_subdirectory(Code)
if(BUILD_TESTING)
  include_directories(${CMAKE_SOURCE_DIR}/Testing/)
  add_subdirectory(Testing)
endif()
```

### CMake - Header Only

* Options are:
    * Check small/medium size project into your project
    
* For example:
```
CMakeCatchTemplate/3rdParty/libraryA/version1/Class1.hpp
CMakeCatchTemplate/3rdParty/libraryA/version1/Class2.hpp
```
* Add to CMakeLists.txt
```
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/libraryA/version1/)
```
* You'd have audit trail (via git repo) of when updates to library were made.


### CMake - Header Only External

* If larger, e.g. Eigen

```
C:\3rdParty\Eigen
C:\build\MyProject
C:\build\MyProject-build
```
* Add to CMakeLists.txt
```
include_directories("C:\3rdParty\Eigen\install\include\eigen3\
```
* Hard-coded, but usable if you write detailed build instructions
* Not very platform independent
* Not very flexible
* Can be self contained if you have a Meta-build


### CMake - find_package

* For example:
```
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND ALL_THIRD_PARTY_LIBRARIES ${OpenCV_LIBS})
  add_definitions(-DBUILD_OpenCV)
```
* So a package can provide information on how you should use it
* If its written in CMake code, even better!


### find_package - Intro

* CMake comes with scripts to include 3rd Party Libraries
* You can also write your own ones


### find_package - Search Locations

A Basic Example (for a full example - see docs)

* Given:
```
  find_package(SomeLibrary [REQUIRED])
```

* CMake will search 
    * all directories in CMAKE_MODULE_PATH
    * for FindSomeLibrary.cmake
    * case sensitive
    
    
### find_package - Result

* If file is found, CMake will try to run it, to find that library.
* FindSomeLibrary.cmake should return SomeLibrary_FOUND:BOOL=TRUE if library was found
* Sets any other variables necessary to use the library
* Check CMakeCache.txt


### find_package - Usage

* So many 3rd party libraries are CMake ready.
* If things go wrong, you can debug it - not compiled


### find_package - Fixing

* You can provide patched versions
* Add your source/build folder to the CMAKE_MODULE_PATH
```
set(CMAKE_MODULE_PATH
    ${CMAKE_SOURCE_DIR}/CMake
    ${CMAKE_BINARY_DIR}
    ${CMAKE_MODULE_PATH}
   )
```
* So CMake will find your version before the system versions


### find_package - Tailoring

* You can write your own
    * e.g. FindEigen in [CMakeCatchTemplate](https://github.com/MattClarkson/CMakeCatchTemplate/blob/master/CMake/FindEigen.cmake)
* Use CMake to substitute variables
* Force include/library dirs
* Useful for vendors API that isn't CMake compatible


### Provide Build Flags

* If a package is found, you can add compiler flag.
```
add_definitions(-DBUILD_OpenCV)
```
* So, you can optionally include things:
```
#ifdef BUILD_OpenCV
#include <cv.h>
#endif
```
* Best not to do too much of this.
* Useful to provide build options, e.g. for running on clusters
 

### Check Before Committing

* Before you commit code to git, 
* Make sure you are compiling what you think you are!
```
#ifdef BUILD_OpenCV
blah blah
#include <cv.h>
#endif
```
* should fail compilation