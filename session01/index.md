---
title: C++ Recap
---

## Aim for Today

* Reminder of 
    * C++ concepts (MPHYGB24)
    * Reminder CMake usage (MPHYGB24)
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
    * Quickly gets out of hand as program size increases
    * Doesn't capture relationships between bits of data
    * Can't control access to data
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
    
### Hello World

This code:

{{cppfrag('01','hello/hello.cc')}}

When built with this CMake file:

{{cmakefrag('01','hello')}}

Produces this output when run:

{{execute('01','hello/hello')}}
