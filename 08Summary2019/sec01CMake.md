---
title: C++ Builder
---

## CMake - cross-platform building


### Cross-platform is common

* Researchers deal with small prototypes
* Develop on Windows/Mac, deploy to Linux cluster
* i.e. cross-platform is common
* Researchers have little programming support


### CMake summary

* CMake generates Makefiles / Visual Studio solution files.
* ```make``` or Visual Studio program runs your compiler
* CMake is NOT compiling your code

So: Learn 1 build system (CMake), and general build code for each platform (Windows, Linux, Mac)


### Going Forward

* There is a CMake book, but its outdated
* Learn by example
* On this course, I gave as much CMake as I could
* But look at commit log for [CMakeTemplate][CMakeTemplate], as I'm still hacking CMake stuff myself.


### Resources

* [CMake homepage][CMakeHome]
* [CMake hello world][CMakeHelloWorld]
* [CMake library and app][CMakeLibraryAndApp]
* [CMake with Catch2][CMakeCatch2]
* [CMakeTemplate with SuperBuild / MetaBuild][CMakeTemplate] and the associated [CMakeTemplateRenamer][CMakeTemplateRenamer] to generate new projects.

[CMakeHome]: https://cmake.org/
[CMakeHelloWorld]: https://github.com/MattClarkson/CMakeHelloWorld
[CMakeLibraryAndApp]: https://github.com/MattClarkson/CMakeLibraryAndApp
[CMakeCatch2]: https://github.com/MattClarkson/CMakeCatch2
[CMakeTemplate]: https://github.com/MattClarkson/CMakeCatchTemplate
[CMakeTemplateRenamer]: https://github.com/MattClarkson/CMakeTemplateRenamer
