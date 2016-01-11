---
title: Object Oriented Review
---

{% idio cpp %}

## Object Oriented Review

### C-style Programming

* Procedural programming
* Pass data to functions

{% code snippets/callFunction.cc %}


### Disadvantages

* Can get out of hand as program size increases
* Can't easily describe relationships between bits of data
* Can't easily control access to data


### C Struct

* So, in C, the struct was invented
* Basically a class without methods

{% code snippets/struct.cc %}


### C++ Class

* C++ provides the class to enhance the language with user defined types
* Once defined, use types as if native to the language


### Abstraction

* C++ class mechanism enables you to define a type
    * independent of its data
    * independent of its implementation
    * class defines concept or blueprint
    * instantiation creates object


### Class Example

{% code snippets/abstraction.cpp %}


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



{% endidio %}
