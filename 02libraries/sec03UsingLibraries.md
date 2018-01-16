---
title: Using Libraries
---

## Using libraries

### Where from?

* Package Manager (Linux/Mac)
    * Precompiled
    * Stable choice
    * Inter-dependencies work
* Linux
    * ```sudo apt-get install```
    * ```sudo yum install```
* Mac    
    * ```sudo port install```
    * ```brew install```
    

### Windows

* Libraries typically:
    * Randomly installed location
    * In system folders
    * In developer folders
    * In build folder
* Try [Chocolatey](http://chocolatey.org) package manager


### Package Managers

* So, if you can use standard versions of 3rd party libraries
* Package managers are a good way to go
* You just need to specify what versions so your collaborator can check


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

We setup ```MyProject-build``` to know the location of ITK and VTK install folder.
    
    
### Meta-Build / Super-Build

* 2 basic approaches
    * Meta-Build, a.k.a SuperBuild
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

We setup ```MyProject-build``` to know the location of ITK and VTK that it itself compiled.

### Pro's / Con's

* External Build
    * Pro's - build each dependency once
    * Con's - collaborators will do this inconsistently
    * Con's - how to manage multiple versions of all dependencies
* Meta Build
    * Pro's - all documented, all self-contained, easier to share
    * Con's - Slow build? Not a problem.
