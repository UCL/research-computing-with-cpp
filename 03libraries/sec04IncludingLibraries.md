---
title: Linking Libraries
---

## Linking libraries

### Aim

* Can be difficult to include a C++ library
* Step through
    * Dynamic versus Static linking
    * Ways of setting paths/library names
        * Platform specific 
        * Use of CMake
    * Packaging - concepts only
* Aim for - source code based distribution


### Recap

* So far
    * Seen how to chose
    * Static / Dynamic
    * How to structure layout
* Now we want to use a library


### Fundamental

* Access to header files - declaration
* Access to compiled code - definition


### Linux/Mac

```
g++ -c -I/users/me/myproject/include main.cpp
g++ -o main main.o -L/users/me/myproject/lib -lmylib
```

* ```-I``` to specify include folder
* ```-L``` to specifiy library folder
* ```-l``` to specify the actual library

Notice: you don't specify .a or .so/.dylib if only 1 type exists.


### Windows

* Visual Studio (check version)
* Project Properties
    * C/C++ -> Additional Include Directories.
    * Configuration Properties -> Linker -> Additional Library Directories
    * Linker -> Input -> Additional Dependencies.
* Check compile line - its equivalent to Linux/Mac, -I, -L, -l


### Use of CMake

* Several ways depending on your setup
    * Hard code paths and library names
    * Use a FindModule
    * Generate and use a FindModule





