---
title: C++ Recap
---

## Aim

* Reminder of 
    * C++ concepts ([MPHYGB24][MPHYGB24])
    * CMake usage ([MPHYGB24][MPHYGB24])
* Get started with some scaffold code
* Unit Testing concepts
* Unit Testing in practice
  
From the outset we encourage Test Driven Design (TDD).
We want good (reliable, reproducible) code.
  
## MPHYGB24

* Lecture 4: Compiling a library, testing debugging
* Lecture 5: Arrays
* Lecture 6: Structures, dynamically allocated arrays
* Lecture 7: Classes
* Lecture 8: Operator overloads, inheritance
* Lecture 9: Polymorphism

## Classes
 
* Procedural programming: pass data to functions
    * Can get out of hand as program size increases
    * Can't easily describe relationships between bits of data
    * Can't easily control access to data
* Object oriented programming: describe types and how they interact

## Abstraction

* Enables you to define a type
    * Class defines concept or "blueprint"
    * We "instantiate" to create a specific object 
* Example: Fraction data type
{{cppfrag('01','fraction/fraction.h')}}

## Encapsulation

* Encapsulation is:
    * Bundling together methods and data
    * Restricting access, defining public interface
* For class methods/variables:
    * `private`: only available in this class
    * `protected`: available in this class and derived classes
    * `public`: available to anyone with access to the object
    
## Inheritance

* Used for:
    * Defining new types based on a common type
    * Reduce code duplication, less maintenance
* Careful:
    * Types in a hierarchy MUST be related
    * Don't over-use this, just to save code duplication
    * There are other ways 
* Example: Shapes
{{cppfrag('01','shape/shape.h')}}


## Polymorphism

* Several types:
    * "subtype": via inheritance
    * "parametric": via templates
    * "ad hoc": via function overloading
* In C++, normally we refer to "subtype" polymorphism
* Is the provision of a common interface to entities of different types
* Example: Shape
{{cppfrag('01','shape/shapeTest.cc')}}

## Further Reading

* Every C++ developer should keep reading
    * [Effective C++][Meyers], Meyers
    * [More Effective C++][Meyers], Meyers
    * [Effective STL][Meyers], Meyers
    * Design Patterns (1994), Gamma, Help, Johnson and Vlassides

## Practical Tips
* If:
    * More coding, more things go wrong
    * Everything gets messy
    * Feeling that you're digging a hole
* Pragmatic tips as how to do this in practice
    * In a scientific research sense

## Coding tips
* Follow coding conventions for your project 
* Compile often
* Version control
    * Commit often
    * Useful commit messages - don't state what can be diff'd, explain why.
    * Short running branches
    * Covered on [MPHYG001][MPHYG001]    
* Class: "does exactly what it says on the tin"
* Class: build once, build properly, testing is key.

## C++ tips
Numbers in brackets refer to Scott Meyers "Effective C++" book.

* Declare data members private (22)
* Initialise objects properly. Throw exceptions from constructors. (4) 
* Use `const` whenever possible (3) 
* Make interfaces easy to use correctly and hard to use incorrectly (18) 
* Prefer non-member non-friend functions to member functions (better encapsulation) (23) 
* Avoid returning "handles" to object internals (28) 
* Never throw exceptions from destructors

## OO tips
* Make sure public inheritance really models "is-a" (32) 
* Learn alternatives to polymorphism (Template Method, Strategy) (35) 
* Model "has-a" through composition (38) 
* Understand [Dependency Injection][DependencyInjection].
* i.e. most people overuse inheritance

## Scientific Computing tips
* Papers require numerical results, graphs, figures, concepts
* Optimise late
    * Correctly identify tools to use
    * Implement your algorithm of choice
    * Provide flexible design, so you can adapt it and manage it
    * Only optimise the bits that are slowing down the production of interesting results
* So, this course will provide you with an array of tools

## CMake
* This is a practical course
* We need to run code
* Use CMake as a build tool
* CMake produces
    * Windows: Visual Studio project files
    * Linux: Make files
    * Mac: XCode projects, Make files
* Our code will provide CMake code and boiler plate code

## CMake Usage
Typically, to do an "out-of-source" build
```
cd ~/myprojects
git clone http://github.com/somecode
mkdir somecode-build
cd somecode-build
cmake ../somecode
make
```
    
# Unit Testing

## Hello World
#This code:
#
#{{cppfrag('01','hello/hello.cc')}}
#
#When built with this CMake file:
#{{cmakefrag('01','hello')}}
#
#Produces this output when run:
#
#{{execute('01','hello/hello')}}

[MPHYGB24]: https://moodle.ucl.ac.uk/course/view.php?id=5395
[Meyers]: http://www.aristeia.com/books.html
[MPHYG001]: https://moodle.ucl.ac.uk/course/view.php?id=28759
[DependencyInjection]: http://en.wikipedia.org/wiki/Dependency_injection