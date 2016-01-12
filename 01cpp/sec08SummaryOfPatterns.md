---
title: RAII Pattern
---

{% idio cpp %}

## RAII Pattern

### What is it?

* Resource Allocation Is Initialisation (RAII)
* Obtain all resources in constructor
* Release them all in destructor
* Scott Meyers ?


### Why is it?

* Guaranteed fully initialised object
* Objects on stack are guaranteed to be destroyed when an exception is thrown and stack is unwound
    * Including smart pointers to objects


### Example




{% endidio %}
