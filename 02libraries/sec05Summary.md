---
title: Summary
---

## Summary

### No Magic Answer

* C++ is more tricky than MATLAB / Python
* While package managers make it easier, you still need to understand what you're building.


### A Few Good Libraries

Main advice for libraries:

* As few as possible
* As high quality as possible


### Ways to Use

* Easiest - header only
    * just include directly in your source code
    * use cmake ```include_directories()```
    * compile it in
* use packages from package manager
    * use cmake to ```find_package```
    * set variables to your installed version
* use build system to build everything
    * you control ALL flags
    * cmake does ```find_package``` on things you just compiled


### Git Submodule Anyone?

* git submodule
    * If all dependencies are cmake'd
    * Put each project in a git submodule
    * Cmake can configure whole project
    * Not really used on this course