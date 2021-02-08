---
title: Examples
---

## Examples

Let's put in practice all we've learnt about libraries using CMake.

Remember, fundamentally, we need:

* Access to header files - declarations
* Access to compiled code - definitions

But we also need to take care of all this:

* Static/Dynamic
* Package Managers / Build your own
* External build / Internal Build
* Release / Debug
* `-I`, `-L`, `-l` flags

Therefore, header only libraries are very attractive!

## Header Only

### CMake - Header next to code

Check the [CMakeCatch2 template repository][gh-catch].
There we've include [catch2 test-framework library][catch2] v2.1.2 with the `catch.hpp` header only
file in [`CMakeCatch2/Testing/catch.hpp`][gh-catch-hpp].

This is then include using CMake in `CMakeCatch2/CMakeLists.txt`, from [line 201][gh-cmake-catch]

```cmake
if(BUILD_TESTING)
  include_directories(${CMAKE_SOURCE_DIR}/Testing/)
  add_subdirectory(Testing)
endif()
```

Which is adding the `Testing` directory as an include path (`-I`) to find the catch2 library
and adds that directory too for the building of the tests available there. In this case, tests will
only be built if the variable `BUILD_TESTING` is enabled.

### CMake - Header under a 3rd Party directory

As with Catch2, imagine that we place the headers of that library under a 3rdParty directory:

```
CMakeCatchTemplate/3rdParty/libraryA/version1/Class1.hpp
CMakeCatchTemplate/3rdParty/libraryA/version1/Class2.hpp
```

Then, we can add these to CMake as

```cmake
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/libraryA/version1/)
```

Keeping the repository under a version control system (e.g., git) this will give us an
audit trail of when updates to that external library were made.


### CMake - Header on a external location

If you think a library is too large to keep it between your repository you may keep it
in an external location.

For example, here we have [Eigen][eigen] on a directory outside our project:

```
C:\3rdParty\Eigen
C:\build\MyProject
C:\build\MyProject-build
```

On the `CMakeLists.txt` we could refer to it as:

```cmake
include_directories("C:\3rdParty\Eigen\install\include\eigen3")
```

This has some problems:

* Hard-coded path, but still usable if you write detailed build instructions
* Not platform independent
* Not very flexible

Let's see what other options we have.

### CMake - Using `find_package`

In general, in CMake, each dependency requires a bit of code to look up include
directories, and libraries.

For example to add [OpenCV][opencv] using `find_package`:

```cmake
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND ALL_THIRD_PARTY_LIBRARIES ${OpenCV_LIBS})
add_definitions(-DBUILD_OpenCV)
```

So a 3rd party package can provide information on how you should use it.


#### Types of `find_package`

[`find_package`][find_package] has several different modes:

* *config* mode: Use 3rd party projects own config, e.g., `VTKConfig.cmake`
* *module* mode: Use a `FindModule`, some come with CMake
* *module* mode: Write your own `FindModule`
* *module* mode: Write your own `FindModule` with generated / substituted variables


#### `find_package` - Basic example

This is a basic explanation (for a full example - see [`find_pacage` documentation][find_package])

When we invoke `find_package` as

```cmake
find_package(SomeLibrary [REQUIRED])
```

CMake will search

* all directories in `CMAKE_MODULE_PATH`
* for `SomeLibraryConfig.cmake` - does *config* mode - and
* for `FindSomeLibrary.cmake` - does *module* mode

Note that CMake is case sensitive!


`find_package(SomeLibrary)` should return `SomeLibrary_FOUND:BOOL=TRUE` if that
library was found. It will also set any other variables necessary to use that library.
You can check CMakeCache.txt to see the result.

Many 3rd party libraries are CMake ready. But if things go wrong, you can debug
it - CMake is all text based.


#### `find_package` - Fixing

You can provide patched versions to the libraries (maybe you want to apply a
patch that hasn't been released yet). To do so, add your source/build folder to
the `CMAKE_MODULE_PATH`:

```cmake
set(CMAKE_MODULE_PATH
    ${CMAKE_SOURCE_DIR}/CMake
    ${CMAKE_BINARY_DIR}
    ${CMAKE_MODULE_PATH}
   )
```

This way, CMake will find your version before the system version.


#### `find_package` - Tailoring

You can write your own instruction for `find_package`. For that you need to write a
`FindSomeLibrary.cmake`. For example, check `FindEigen` in [CMakeCatchTemplate][gh-CatchTemplate-FindEigen].

With this we use CMake to substitute variables and force include/library directories.
It's useful for vendors API that isn't CMake compatible and
for meta-build (It forces directories to match the package you just compiled).


#### Provide Build Flags

When a package is found, you can add compiler flags with

```cmake
add_definitions(-DBUILD_OpenCV)
```

So, you can then optionally include things like:

```cpp
#ifdef BUILD_OpenCV
#include <cv.h>
#endif
```

Note, however, that's best not to do too much of this.
It's useful to provide build options, e.g., for running on clusters.

Always, before you commit code to git,
make sure you are compiling what you think you are!

This should fail compilation:

```cpp
#ifdef BUILD_OpenCV
blah blah
#include <cv.h>
#endif
```

### CMake and a CMake package manager

It's worth mentioning the available CMake package managers, which will try to
download, build and set the 3rd Party libraries to your project.

For example check how you can add Eigen to your project either using
[Hunter][hunter-eigen] or [CPM][cpm-eigen].

## Summary

In short:

* `include_directories()` generates `-I`
* `link_directories()` generates `-L`
* `target_link_libraries(mylibrary PRIVATE ${libs})` generates `-l` for each library

It might not need `link_directories()` if libraries fully qualified.

Try default CMake `find_package` or write your own and add location to `CMAKE_MODULE_PATH`.


[catch2]: https://github.com/catchorg/Catch2
[gh-catch]: https://github.com/UCL/CMakeCatch2
[gh-catch-hpp]: https://github.com/UCL/CMakeCatch2/blob/master/Testing/catch.hpp
[gh-cmake-catch]: https://github.com/UCL/CMakeCatch2/blob/master/CMakeLists.txt#L201
[eigen]: https://eigen.tuxfamily.org/index.php
[opencv]: https://docs.opencv.org/4.5.0/db/df5/tutorial_linux_gcc_cmake.html
[find_package]: https://cmake.org/cmake/help/latest/command/find_package.html
[gh-CatchTemplate-FindEigen]: https://github.com/MattClarkson/CMakeCatchTemplate/blob/master/CMake/FindEigen.cmake
[hunter-eigen]: https://hunter.readthedocs.io/en/latest/packages/pkg/Eigen.html#pkg-eigen
[cpm-eigen]: https://github.com/TheLartians/CPM.cmake/wiki/More-Snippets#Eigen
