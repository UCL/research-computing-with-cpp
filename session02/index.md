---
title: Templates
---

## Why Templates?

### What Are Templates?

* The broader concept is [Generic Programming](http://en.wikipedia.org/wiki/Generic_programming).
    * Write code, where 'type' is provided later
    * Types instantiated at compile time, as they are needed
    * (Remember, C++ is strongly typed)
    
    
### Example
    
You probably use them already. Example type (class):

```c++
std::vector<int> myVectorInts;
```

Example algorithm: [C++ sort](http://www.cplusplus.com/reference/algorithm/sort/)

```c++
std::sort(myVectorInts.begin(), myVectorInts.end());
```

Aim: Write functions, classes, in terms of future types.


### Why Are Templates Useful?

* Generic programming:
    * without pre-processor macros
    * so, maintaining type safety
    * separate algorithm from implementation
    * extensible, optimisable via Template Meta-Programming (TMP)


### Why Templates in Research?

* Generalise 2D, 3D, n-dimensions, (e.g. [ITK](http://www.itk.org) )
* Test numerical code with simple types, apply to complex/other types
* Several useful libraries for research


### Are Templates Difficult?

* Some say: notation is ugly
    * Does take getting used to
    * Use ```typedef``` to simplify
* Some say: verbose, confusing error messages
    * Nothing intrinsically difficult
    * Take small steps, compile regularly
    * Learn to think like a compiler
* Code infiltration
    * So, use sparingly
    
    
### Why Teach Templates?

* More common in research code, than business code
* In research, more likely to 'code for the unknown'
* Standard Template Library uses them
* Boost, Qt, EIGEN uses them
* Benchmark of "intermediate" programmer?
