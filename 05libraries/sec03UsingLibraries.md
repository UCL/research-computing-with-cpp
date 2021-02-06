---
title: Using Libraries
---

## Using libraries

### Package Managers

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
* Please clean your machine!!
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
        

### Build Your Own

* 2 basic approaches
    * External / Individual build
        * Build dependencies externally
        * Point your software at those packages
    * SuperBuild / Meta-Build
        * Write code to download all dependencies
        * The correct version numbers is stored in code
        

### External / Individual Build

For example

```
C:\build\ITK-v1
C:\build\ITK-v1-build
C:\build\ITK-v1-install
C:\build\VTK-v2
C:\build\VTK-v2-build
C:\build\VTK-v2-install
C:\build\MyProject
C:\build\MyProject-build
```

We setup ```MyProject-build``` to know the location of ITK and VTK install folder.
       

### Meta-Build / Super-Build

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
    * Con's - Slow build? Not a problem if you only run ```make``` in sub-folder ```MyProject-build```
