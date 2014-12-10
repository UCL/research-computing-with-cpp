---
title: C++ Recap
---

## Aim for Today

* Reminder of C++ concepts (MPHYGB24)
* CMake primer (MPHYGB24)
* Scaffold code
* Unit Testing concepts
* Unit Testing in practice
  
From the outset we encourage Test Driven Design (TDD).
  
## Topics from MPHYGB24

* 4 Compiling a library, testing debugging
* 5 Arrays
* 6 Structures, dynamically allocated arrays
* 7 Classes
* 8 Operator overloads, inheritance
* 9 Polymorphism

## Classes
 
* Procedural programming: pass data to functions
* Quickly gets out of hand as program size increases
* Doesn't capture relationships between bits of data
* Can't control access to data
* Object oriented programming: describe types and how they interact

### Hello World

This code:

{{cppfrag('01','hello/hello.cc')}}

When built with this CMake file:

{{cmakefrag('01','hello')}}

Produces this output when run:

{{execute('01','hello/hello')}}
