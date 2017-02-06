---
title: Using Libraries
---

## Using libraries

### Aim

* Can be difficult to include a C++ library
* Step through
    * Dynamic versus Static linking
    * Ways of setting paths/library names
        * Platform specific 
        * Use of CMake
    * Packaging - concepts only
* Aim for - source code based distribution


### Where from?

* Package Manager (Linux/Mac)
    * Precompiled
    * Stable choice
    * Inter-dependencies work
* For example
    * ```sudo apt-get install```
    * ```port install```
    * ```brew install```
    

### Windows

* Libraries typically:
    * Randomly installed location
    * In system folders
    * In developer folders
    * In build folder
* Package managers forthcoming


### Problems

* As soon as you hit a bug in a library
    * How to update?
    * Knock on effects
        * Cascading updates
        * Inconsistent development environment
        

### External Build

* 2 basic approaches
    * Separate build
        * Build dependencies externally
        * Point your software at those packages


### Example

For example

```
C:\build\ITK
C:\build\ITK-build
C:\build\ITK-install
C:\build\VTK
C:\build\VTK-build
C:\build\VTK-install
C:\build\MyProject
C:\build\MyProject-build
```

We setup ```MyProject-build``` to know the location of ITK and VTK. 
    
    
### Meta-Build

* 2 basic approaches
    * Separate build
        * Build dependencies externally
        * Point your software at those packages
    * Meta-Build
        * Your software coordinates building dependencies
        

### Example

For example

```
C:\build\MyProject
C:\build\MyProject-SuperBuild\ITK\src
C:\build\MyProject-SuperBuild\ITK\build
C:\build\MyProject-SuperBuild\ITK\install
C:\build\MyProject-SuperBuild\VTK\src
C:\build\MyProject-SuperBuild\VTK\build
C:\build\MyProject-SuperBuild\VTK\install
C:\build\MyProject-SuperBuild\MyProject-build
```

We setup ```MyProject-build``` to know the location of ITK and VTK. 
