---
title: Next Steps
---

{% idio cpp %}

## Next Steps

### Real-World C++

* Basic C++ course not sufficient
* For all but a few classes, rapidly get tied in knots
* Here we learn:
    * Some C++ features
    * Some general tips
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
* So smart pointers from boost may be in your compiler
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


### Exception Handling

* Exceptions are the C++ or Object Oriented way of Error Handling
* Read [this](https://msdn.microsoft.com/en-us/library/hh279678.aspx) example


### Error Handling C-Style

{% code snippets/errorHandlingInC.cc %}


### Outcome

* Can be perfectly usable
* Depends on depth of function call stack
* Depends on complexity of program
* If deep/large, then becomes unweildy


### Error Handling C++ Style

* Code that throws does not worry about the catcher
* More suited to larger libraries
* (Think about software in layers)
* Exceptions are classes, can carry data
* Exceptions can form class hierarchy


### Practical Tips For Exception Handling

* Decide on error handling strategy at start
* Create your own base class
* Derive all your exceptions from that base class
* Stick to a few obvious classes, not one class for every single error

### More Practical Tips For Exception Handling

* Look at [C++ standard classes](http://www.cplusplus.com/reference/exception/) and [tutorial](http://www.cplusplus.com/doc/tutorial/exceptions/)
* An exception macro may be useful, e.g. [mitk::Exception](https://github.com/MITK/MITK/blob/master/Modules/Core/include/mitkException.h) and [mithThrow()](https://github.com/MITK/MITK/blob/master/Modules/Core/include/mitkExceptionMacro.h)
* Beware side-effects
    * Perform validation before updating any member variables


### Construction

* What could be wrong with this:

{% code snippets/constructorDependency.cc %}


### Unwanted Dependencies

* If constructor instantiates class directly:
    * Hard-coded class name
    * Duplication of initialisation code

### Dependency Injection

* Read [Inversion of Control Containers and the Dependency Injection Pattern](http://www.martinfowler.com/articles/injection.html)
* Type 2 - Constructor Injection

{% code snippets/constructorInjection.cc %}





{% endidio %}
