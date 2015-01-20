---
title: Object Oriented Review
---

## Object Oriented Review

### Classes
 
* With procedural programming
    * pass data to functions
    * can get out of hand as program size increases
    * can't easily describe relationships between bits of data
    * can't easily control access to data
* With object oriented programming
    * describe types and how they interact
* Once defined 
    * use types as if native to the language 


### Abstraction

* C++ class mechanism enables you to define a type
    * independent of its data 
    * independent of its implementation
    * class defines concept or blueprint
    * instantiation creates object 
* Example: Fraction data type
{{cppfrag('01','fraction/fraction.h')}}


### Encapsulation

* Encapsulation is:
    * Bundling together methods and data
    * Restricting access, defining public interface
* For class methods/variables:
    * `private`: only available in this class
    * `protected`: available in this class and derived classes
    * `public`: available to anyone with access to the object
    
    
### Inheritance

* Used for:
    * Defining new types based on a common type
    * Reduce code duplication, less maintenance
* Careful:
    * Types in a hierarchy MUST be related
    * Don't over-use this, just to save code duplication
    * There are other ways 
* Example: Shapes
{{cppfrag('01','shape/shape.h')}}


### Polymorphism

* Several types:
    * (normally) "subtype": via inheritance
    * "parametric": via templates
    * "ad hoc": via function overloading
* Common interface to entities of different types
* Same method, different behaviour
* Example: Shape
{{cppfrag('01','shape/shapeTest.cc')}}


