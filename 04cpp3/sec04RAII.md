---
title: RAII Pattern
---

{% idio cpp %}

## RAII Pattern

### What is it?

* [Resource Allocation Is Initialisation (RAII)](https://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization)
* Obtain all resources in constructor
* Release them all in destructor


### Why is it?

* Guaranteed fully initialised object once constructor is complete
* Objects on stack are guaranteed to be destroyed when an exception is thrown and stack is unwound
    * Including smart pointers to objects


### Example

* You may already be using it: [STL example](https://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization)
* [Another example](https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Resource_Acquisition_Is_Initialization)

### Homework 19

* Create a simple class `Foo` that contains a data member that is a raw pointer `bptr` to another class `Bar` that contains an integer as a data member 
* Add a `std::cout` to the constructor and destructor of both classes so that you know when they have been called
* Implement the RAII pattern to create and destroy the `Bar` object that  `bptr` points to 
* Create an instance of `Foo foo` in your application and confirm that if an exception is thrown before Foo goes out of scope that the destructor for both `Foo` and `Bar` are called and the `Bar` object is released

{% endidio %}
