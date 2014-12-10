---
title: C++ Recap
---

## Aim for Today

* Reminder of 
    * C++ concepts ([MPHYGB24][MPHYGB24])
    * Reminder CMake usage ([MPHYGB24][MPHYGB24])
* Get started with some scaffold code
* Unit Testing concepts
* Unit Testing in practice
  
From the outset we encourage Test Driven Design (TDD).
  
## Topics covered in MPHYGB24

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

## Classes - Abstraction

* Enables you to define a type
    * Class defines concept or "blueprint"
    * We "instantiate" to create a specific object 
* Example: Fraction data type
{{cppfrag('01','fraction/fraction.h')}}

## Classes - Encapsulation

* Encapsulation is:
    * Bundling together methods and data
    * Restricting access, defining public interface
* For class methods/variables:
    * `private`: only available in this class
    * `protected`: available in this class and derived classes
    * `public`: available to anyone with access to the object
    
## Classes - Inheritance

* Used for:
    * Defining new types based on a common type
    * Reduce code duplication, less maintenance
* Careful:
    * Types in a hierarchy MUST be related
    * Don't over-use this, just to save code duplication
    * There are other ways 
* Example: Shapes
{{cppfrag('01','shape/shape.h')}}


## Classes - Polymorphism

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
    
       
### Hello World

This code:

{{cppfrag('01','hello/hello.cc')}}

When built with this CMake file:

{{cmakefrag('01','hello')}}

Produces this output when run:

{{execute('01','hello/hello')}}

[MPHYGB24]: https://moodle.ucl.ac.uk/course/view.php?id=5395
[Meyers]: http://www.aristeia.com/books.html