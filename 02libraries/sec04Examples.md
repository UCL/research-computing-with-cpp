---
title: Examples
---

## Examples


### Reminder

* Fundamentally, we need:
    * Access to header files - declaration
    * Access to compiled code - definition


### Header Only?

* After all this:
    * Static/Dynamic
    * Package Managers / Build your own
    * External build / Internal Build
    * Release / Debug
    * -I, -L, -l
* Header only libraries are very attractive.


### CMake - Example

* catch.hpp Header files in project [CMakeCatch2](https://github.com/MattClarkson/CMakeCatch2)
* From ```CMakeCatch2/CMakeLists.txt```

```
include_directories(${CMAKE_SOURCE_DIR}/Code/)
add_subdirectory(Code)
if(BUILD_TESTING)
  include_directories(${CMAKE_SOURCE_DIR}/Testing/)
  add_subdirectory(Testing)
endif()
```


### CMake - Header Only

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


### CMake - Header Only, External

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


### CMake - find_package

* For example:
```
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND ALL_THIRD_PARTY_LIBRARIES ${OpenCV_LIBS})
  add_definitions(-DBUILD_OpenCV)
```
* So a 3rd party package can provide information on how you should use it
* If its written in CMake code, even better!


### Use of CMake

* Use ```find_package```
    * Use 3rd party projects own config, eg. ```VTKConfig.cmake```
    * Use a FindModule, some come with CMake
    * Write your own FindModule
    * Write your own FindModule with generated / substituted variables


### find_package - Intro

* CMake comes with scripts to include various 3rd Party Libraries
* 3rd Parties can write these scripts aswell
* You can also write your own scripts


### find_package - Search Locations

A Basic Example (for a full example - see docs)

* Given:
```
  find_package(SomeLibrary [REQUIRED])
```

* CMake will search 
    * all directories in CMAKE_MODULE_PATH
    * for SomeLibraryConfig.cmake - does *config* mode
    * for FindSomeLibrary.cmake - does *module* mode
    * case sensitive
    
    
### find_package - Result

* ```find_package(SomeLibrary)``` should return SomeLibrary_FOUND:BOOL=TRUE if library was found
* Sets any other variables necessary to use the library
* Check CMakeCache.txt to see result


### find_package - Usage

* So many 3rd party libraries are CMake ready.
* If things go wrong, you can debug it - CMake is all text based.


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
    * Useful for meta-build. Force directories to match the package you just compiled.


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


### Summary

* Basic aim:
    * ```include_directories()``` generates -I
    * ```link_directories()`` generates -L
    * ```target_link_libraries(mylibrary PRIVATE ${libs})``` generates -l for each library
* Might not need ```link_directories()``` if libraries fully qualified
* Try default CMake find_package
* Or write your own and add location to CMAKE_MODULE_PATH
