---
title: Templates
---

## Using Templates

### What Are Templates?

* C++ templates allow functions/classes to operate on generic types.
* See: [Generic Programming](http://en.wikipedia.org/wiki/Generic_programming).
* Write code, where 'type' is provided later
* Types instantiated at compile time, as they are needed
* (Remember, C++ is strongly typed)


### You May Already Use Them!

You probably use them already. Example type (class):

```
std::vector<int> myVectorInts;
```

Example algorithm: [C++ sort](http://www.cplusplus.com/reference/algorithm/sort/)

```
std::sort(myVectorInts.begin(), myVectorInts.end());
```

Aim: Write functions, classes, in terms of future/other/generic types, type provided as parameter.


### Why Are Templates Useful?

* Generic programming:
    * not pre-processor macros
    * so maintain type safety
    * separate algorithm from implementation
    * extensible, optimisable via [Template Meta-Programming](97TemplateMetaProg) (TMP)


### Book

* You should read ["Modern C++ Design"](http://erdani.com/index.php/books/modern-c-design/)
* 2001, but still excellent text on templates, meta-programming, policy based design etc.
* This section of course, gives basic introduction for research programmers


### Are Templates Difficult?

* Some say: notation is ugly
    * Does take getting used to
    * Use ```typedef``` to simplify
* Some say: verbose, confusing error messages
    * Nothing intrinsically difficult
    * Take small steps, compile regularly
    * Learn to think like a compiler
* Code infiltration
    * Use sparingly
    * Hide usage behind clean interface


### Why Templates in Research?

* Generalise 2D, 3D, n-dimensions, (e.g. [ITK](http://www.itk.org) )
* Test numerical code with simple types, apply to complex/other types
* Several useful libraries for research


### Why Teach Templates?

* Standard Template Library uses them
* More common in research code, than business code
* In research, more likely to 'code for the unknown'
* Boost, Qt, EIGEN uses them
