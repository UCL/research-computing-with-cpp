---
title: Memory Management 
---

{% idio cpp %}

## Memory Management 

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
 
{% code snippets/unique.cc %}


### Using Smart Pointers

* Varying semantics
    * [unique_ptr](http://en.cppreference.com/w/cpp/memory/unique_ptr) - uniquely owns
    * [shared_ptr](http://en.cppreference.com/w/cpp/memory/shared_ptr) - shares
    * [weak_ptr](http://en.cppreference.com/w/cpp/memory/weak_ptr) - temporary tracker
    * [David Kieras online paper](http://www.umich.edu/~eecs381/handouts/C++11_smart_ptrs.pdf)


### Boost Vs Standard Library

* Boost has become a sandbox for standard C++
* So check your compiler version
* Features may be in your standard compiler
* So smart pointers from boost may be in your compiler under `std::`
    * For [MITK](http://www.mitk.org)/[NifTK](http://www.niftk.org): C++11: gcc 4.7.3, clang 3.4, apple clang 5.0, MSVC 17.0.61030.0 (2012 update 4)


### Intrusive Vs Non-Intrusive

* Intrusive - Base class maintains a reference count eg. [ITK](http://www.itk.org)
* Non-intrusive - Reference count is in the pointer eg. [Boost](http://www.boost.org)

Question what are the implications when passing to a function?


### ITK Smart Pointers

{% code snippets/itkSmartPointer.cc %}


### Conclusion for Smart Pointers

* Lots of other Smart Pointers
    * [Qt Smart Pointers](https://wiki.qt.io/Smart_Pointers)
    * [VTK](http://www.vtk.org/Wiki/VTK/Tutorials/SmartPointers)
* Always read the manual
* Always consistently use it
* Don't be tempted to write your own
* Still easier than raw pointers

{% endidio %}
