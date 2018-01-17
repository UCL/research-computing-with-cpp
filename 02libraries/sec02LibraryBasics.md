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


### Linux/Mac

```
g++ -c -I/users/me/myproject/include main.cpp
g++ -o main main.o -L/users/me/myproject/lib -lmylib
```

* ```-I``` to specify include folder
* ```-L``` to specify library folder
* ```-l``` to specify the actual library


### Compiler switches

So, that means

* Include directory
    * ```-I /some/directory``` 
* Library directory
    * ```-L /some/directory``` 
* Library
    * ```-l library``` 

Similar concept on Windows, Linux and Mac.


### Native Build Platform

* Look inside
    * Makefile
    * Visual Studio options
* Basically constructing ```-I```, ```-L```, ```-l``` switches to pass to command line compiler.        


### Windows Compiler Switches

* Visual Studio (check version)
* Project Properties
    * C/C++ -> Additional Include Directories.
    * Configuration Properties -> Linker -> Additional Library Directories
    * Linker -> Input -> Additional Dependencies.
* Check compile line - its equivalent to Linux/Mac, -I, -L, -l


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


### A Few Good Libraries

Due to all those issues shown above, again, the main advice for libraries:

* As few as possible
* As good as possible