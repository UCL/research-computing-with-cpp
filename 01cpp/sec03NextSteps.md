---
title: Next Steps
---

{% idio cpp %}

## Next Steps

### Real-World C++

* Basic C++ course not sufficient
* For all but a few classes, rapidly get tied in knots
* Here we learn:
    * Some general tips
    * Some C++ features
    * Some workflow
* Try to tie it back to research examples


### C++ Features

* Need to know:
    * Memory, smart pointers
    * Exception handling
    * Construction, dependency injection
    * Mutable/Immutable/Encapsulation
    * Don't overuse inheritance

### Smart Pointers

* ```new/delete``` not good enough
* You will introduce bugs
* So, use smart pointers
* Class that looks like a pointer, but smarter
    * sharing
    * auto-deletion
    * depends on implementation


### Example

* unique_ptr from [here](http://en.cppreference.com/w/cpp/memory/unique_ptr)
 

### Using Smart Pointers

* Varying semantics
    * [unique_ptr](http://en.cppreference.com/w/cpp/memory/unique_ptr)
    * [shared_ptr](http://en.cppreference.com/w/cpp/memory/shared_ptr)
    * [weak_ptr](http://en.cppreference.com/w/cpp/memory/weak_ptr)
    * [David Kieras online paper](http://www.umich.edu/~eecs381/handouts/C++11_smart_ptrs.pdf)


### Boost Vs Standard Library

* Boost has become a `sandbox` for standard C++
* So check your compiler version
* Features may be in your standard compiler
* So smart pointers from boost may be in your compiler
    * C++11: gcc 4.7.3, clang 3.4, apple clang 5.0, MSVC 17.0.61030.0 (2012 update 4)


### Intrusive Vs Non-Intrusive

* Intrusive - Base class maintains a reference count eg. [ITK](http://www.itk.org)
* Non-intrusive - Reference count is in the pointer.

Question what are the implications when passing to a function?


### ITK Smart Pointers

{% code snippets/itkSmartPointer.cc %}


### Conclusion for Smart Pointers

* Lots of other Smart Pointers
    * [Qt Smart Pointers](https://wiki.qt.io/Smart_Pointers)
    * [VTK http://www.vtk.org/Wiki/VTK/Tutorials/SmartPointers]
* Always read the manual
* Always consistently use it
* Don't be tempted to write your own


{% endidio %}
