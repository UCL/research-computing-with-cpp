---
title: Linking Libraries
---

## Linking libraries

### Aim

* Can be difficult to include a C++ library
* Step through
    * [Dynamic versus Static linking](http://www.learncpp.com/cpp-tutorial/a1-static-and-dynamic-libraries/)
    * Ways of setting paths/library names
        * Platform specific 
        * Use of CMake
    * Packaging - concepts only
* Aim for - source code based distribution


### Linking

* Code is split into functions/classes
* Related functions get grouped into libraries
* Libraries have namespaces, names, declaration, definitions (implementations)
* Libraries get compiled - saves compilation time
* End User needs
    * Header file = declarations
    * Object code / library file = implementations
    
    
### Static Linking

* Windows (.lib), Mac/Linux (.a)
* Compiled code from static library is copied into the current translation unit.
* Increases disk space
* Current translation unit then does not depend on that library.


### Dynamic Linking

* Windows (.dll), Mac (.dylib), Linux (.so)
* Compiled code is left in the library.
* At runtime, 
    * OS loads the executable
    * Finds any unresolved libraries
        * Various search mechanisms
    * Recursive process
* Current translation unit has a known dependency remaining.
    
    
### Dynamic Loading

* System call to load a library (dlopen/LoadLibrary)
* Dynamically discover function names and variables
* Execute functions
* Normally used for plugins
* Not covered here


### Space Comparison

* If you have many executables linking a common set of libraries
* Static
    * Code gets copied - each executable bigger
    * Doesn't require searching for libraries at run-time
* Dynamic
    * Code gets referenced - smaller executable
    * Require's finding libraries at run-time
    

### How to Check

* Windows - Dependency Walker
* Linux - ```ldd```
* Mac - ```otool -L```
* (live demo on Mac)


### Note about Licenses

CAVEAT: Again - this is not legal advice.

* Static linking considered to make your work a 'derivative work'
* If you use LGPL - use dynamic linking