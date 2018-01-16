---
title: Library Basics
---

## Library Basics

### Aim

You need your compiler to find:

* Headers: ```#include```
* Library:
    * Dynamic: .dylib, .so, .lib / .dll
    * Static: .a, .lib


### Compiler switches

So, that means

* Include directory
    * ```-I /some/directory``` 
* Library directory
    * ```-L /some/directory``` 
* Library
    * ```-l library``` 

Similar concept on Windows/Linux and Mac.


### Native Build Platform

* Look inside
    * Makefile
    * Visual Studio options
* Basically constructing ```-I```, ```-L```, ```-l``` switches to pass command line compiler.        


### Difficulties with Libraries

* Discuss
* (confession time)


### Location Issues

When you use a library:

* Where is it?
* Header only?
* System version or your version?
* What about bugs? How do I upgrade?


### Compilation Issues

When you use a library:

* Which library version?
* Which compiler version?
* Debug or Release?
* [Static or Dynamic?](http://www.learncpp.com/cpp-tutorial/a1-static-and-dynamic-libraries/)
* 32 bit / 64 bit?
* Platform specific flags?
* Pre-installed, or did you compile it?


### No Magic Answer

While package managers make it easier, you still need to understand what you're building.


### A Few Good Libraries

Again - main advice for libraries:

* As few as possible
* As good as possible