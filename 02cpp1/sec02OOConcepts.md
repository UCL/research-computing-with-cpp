---
title: Object Oriented Review
---

{% idio cpp %}

## Object Oriented Review

### C-style Programming

* Procedural programming
* Pass data to functions


### Function-style Example

{% fragment call_function, snippets/callFunction.cc %}


### Disadvantages

* Can get out of hand as program size increases
* Can't easily describe relationships between bits of data
* Relies on method documentation, function and variable names
* Can't easily control/(enforce control of) access to data


### C Struct 

* So, in C, the struct was invented
* Basically a class without methods
* This at least provides a logical grouping


### Struct Example

{% fragment struct, snippets/struct.cc %}


### C++ Class

* C++ provides the class to enhance the language with user defined types
* Once defined, use types as if native to the language


### Abstraction

* C++ class mechanism enables you to define a type
    * independent of its data
    * independent of its implementation
    * class defines concept or blueprint
    * instantiation creates object


### Abstraction - Grady Booch

“An abstraction denotes the essential characteristics of an object that distinguish it 
from all other kinds of objects and thus provide crisply defined conceptual boundaries, 
relative to the perspective of the viewer.”


### Class Example

{% code snippets/abstraction.cc %}


### Encapsulation

* Encapsulation is:
    * Bundling together methods and data
    * Restricting access, defining public interface
* Describes how you correctly use something


### Public/Private/Protected

* For class methods/variables:
    * `private`: only available in this class
    * `protected`: available in this class and derived classes
    * `public`: available to anyone with access to the object

* (public, protected, private inheritance comes later)


### Class Example

{% code fraction/fraction.h %}


### Inheritance

* Used for:
    * Defining new types based on a common type
* Careful:
    * Beware - "Reduce code duplication, less maintenance"
    * Types in a hierarchy MUST be related
    * Don't over-use inheritance
    * We will cover other ways of object re-use


### Class Example

{% code shape/shape.h %}


### Polymorphism

* Several types:
    * (normally) "subtype": via inheritance
    * "parametric": via templates
    * "ad hoc": via function overloading
* Common interface to entities of different types
* Same method, different behaviour


### Class Example

{% code shape/shapeTest.cc %}

### Homework - 11

* Use [https://github.com/MattClarkson/CMakeCatch2.git](https://github.com/MattClarkson/CMakeCatch2.git) and create a simple shape class for a square with methods to calculate the area
   * Create an object of the class from within an app and print the result of the area calculation to screen
   * Try to write some unit tests to check the class behaves as expected 

### Homework - 12
* Again using `CMakeCatch2` as a basis use inheritance and polymorphism to create a shape base class and a set of derived classes for a square, rectangle
   * Draw a class inheritance diagram before starting
   * Confirm that the get area functions behave differently for the different dervied classes

{% endidio %}
