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


{% endidio %}
