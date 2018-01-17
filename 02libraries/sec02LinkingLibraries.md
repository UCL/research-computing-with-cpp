---
title: Linking Libraries
---

## Linking libraries

### Linking

* From first lecture
    * Code is split into functions/classes
    * Related functions get grouped into libraries
    * Libraries get compiled / archived into one file
* End User needs
    * Header file = declarations (and implementation if header only)
    * Object code / library file = implementations
    
    
### Static Linking

* Windows (.lib), Mac/Linux (.a)
* Compiled code from static library is copied into the current translation unit.
* Increases disk space compared with dynamic linking.
* Current translation unit then does not depend on that library.


### Dynamic Linking

* Windows (.dll), Mac (.dylib), Linux (.so)
* Compiled code is left in the library.
* At runtime, 
    * OS loads the executable
    * OS / Linker finds any unresolved libraries
    * Recursive process
* Saves disk space compared with static linking.
* Faster compilation/linking times?
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
    * Requires finding libraries at run-time
* Space - less of a concern these days


### For Scientists

* Ease of use
* Ease of distribution to collaborators
* Prefer static if possible for ease of deployment


### Packaging

* Packaging large apps takes effort
* hire Research Software Engineers


### How to Check

* Windows - Dependency Walker
* Linux - ```ldd```
* Mac - ```otool -L```
* (live demo on Mac)
