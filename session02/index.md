---
title: C++ Templates
---

## Why Templates?

### What Are Templates?

* The broader concept is [Generic Programming](http://en.wikipedia.org/wiki/Generic_programming).
    * Write code, where 'type' is defined later
    * Types defined at compile time, as they are needed
    * (Remember, C++ is strongly typed)
    
    
### Example
    
Example type (class):

```c++
std::vector<int> myVectorInts;
```

Example algorithm: [C++ sort](http://www.cplusplus.com/reference/algorithm/sort/)

```c++
std::sort(myVectorInts.begin(), myVectorInts.end());
```

Aim: Write functions, classes, in terms of future types.


### Why Are Templates Useful?

* Generic, without pre-processor macros
* Generic, maintaining type safety
* Generic, separate algorithm from implementation
* Optimisation via Template Meta-Programming (advanced)


### Why Are Templates Useful in Research?

* Generalise 2D, 3D, n-dimensions, (e.g. [ITK](http://www.itk.org)
* Test numerical code with simple types, apply to complex types
* Several useful libraries for research


### Are Templates Difficult?

* Some say: notation is ugly
    * We will teach ways to cope/simplify notation
* Some say: verbose, confusing error messages
    * Nothing intrinsically difficult
    * Take small steps, compile regularly
    * Think like a compiler
* Code infiltration
    * Use sparingly
    
    
### Why Teach Templates

* More common in research code, than business code
* More likely to 'code for the unknown'
* Standard Template Library uses them
* Boost, Qt, EIGEN uses them
* Benchmark of "intermediate" programmer?
