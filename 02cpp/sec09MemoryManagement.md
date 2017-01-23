---
title: Smart Pointers
---

{% idio cpp %}

## Memory Management 

### Use of Raw Pointers

* Given a pointer passed to a function

```
   void DoSomethingClever(int *a) 
   {
     // write some code
   }
```

* How do we use the pointer? 
* What problems are there?


### Problems with Raw Pointers

* From ["Effective Modern C++", Meyers, p117](https://www.amazon.co.uk/Effective-Modern-Specific-Ways-Improve/dp/1491903996/ref=sr_1_1?ie=UTF8&qid=1484571499&sr=8-1&keywords=Effective+Modern+C%2B%2B).
    * Single object or array?
    * If you are done, do you destroy it?
    * How to destroy it? Call ```delete``` or some method first: ```a->Shutdown();```
    * ```delete``` or ```delete[]```?
    * How to ensure you delete it once?
    * Is it dangling?


### Smart Pointers

* ```new/delete``` on raw pointers not good enough
* You will introduce bugs
* So, use smart pointers
* Class that looks like a pointer, but smarter
    * automatically delete pointed to object
    * more control over sharing


### Using Smart Pointers

* Varying semantics
    * [std::unique_ptr](http://en.cppreference.com/w/cpp/memory/unique_ptr) - models *has-a* but also unique ownership
    * [std::shared_ptr](http://en.cppreference.com/w/cpp/memory/shared_ptr) - models *has-a* but shared ownership
    * [std::weak_ptr](http://en.cppreference.com/w/cpp/memory/weak_ptr) - temporary reference, breaks circular references
    * [David Kieras online paper](http://www.umich.edu/~eecs381/handouts/C++11_smart_ptrs.pdf)
    * ["Effective Modern C++", Meyers, ch4](https://www.amazon.co.uk/Effective-Modern-Specific-Ways-Improve/dp/1491903996/ref=sr_1_1?ie=UTF8&qid=1484571499&sr=8-1&keywords=Effective+Modern+C%2B%2B)


### Comment on Boost

* Boost has become a sandbox for standard C++
* Boost features become part of standard C++, (different name space)
* So smart pointers from boost may be in your compiler under `std::`
* So check your compiler version
    * e.g. For [MITK](http://www.mitk.org)/[NifTK](http://www.niftk.org): C++11: gcc 4.7.3, clang 3.4, apple clang 5.0, MSVC 17.0.61030.0 (2012 update 4)
* Or you could fall back to boost ones


### Intrusive Vs Non-Intrusive

* Intrusive - Base class maintains a reference count eg. [ITK](http://www.itk.org)
* Non-intrusive
    * ```std::unique_ptr```
    * ```std::shared_ptr```
    * ```std::weak_ptr```

Question what are the implications when passing to a function?


### ITK (intrusive) Smart Pointers

{% code snippets/itkSmartPointer.cc %}


### Conclusion for Smart Pointers

* Default to standard library, check compiler
* Lots of other Smart Pointers
    * [Boost](http://www.boost.org) (use STL).
    * [ITK](http://www.itk.org)
    * [VTK](http://www.vtk.org/Wiki/VTK/Tutorials/SmartPointers)
    * [Qt Smart Pointers](https://wiki.qt.io/Smart_Pointers)
* Don't be tempted to write your own
* Always read the manual
* Always consistently use it

{% endidio %}
