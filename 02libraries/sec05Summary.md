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
* As good as possible


### In order of difficulty?

* Easiest - header only
    * just include directly in your source code
    * use cmake ```include_directories()```
    * compile it in
* use package manager
    * cmake just does ```find_package```
* git submodule
    * If all dependencies are cmake'd
    * Put each project in a git submodule
    * cmake can configure whole project
* use build system to build everything
    * you control ALL flags
    * cmake does ```find_package``` on things you just compiled
