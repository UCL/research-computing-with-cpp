---
title: Error Handling 
---

{% idio cpp %}

## Error Handling

### Exceptions 

* Exceptions are the C++ or Object Oriented way of Error Handling
* Read [this](https://msdn.microsoft.com/en-us/library/hh279678.aspx) example


### Exception Handling Example

{% code snippets/errorHandlingInCPP.cc %}


### What's the Point?

* Have separated error handling logic from application logic
* First, lets look at C-style return codes


### Error Handling C-Style

{% code snippets/errorHandlingInC.cc %}


### Outcome

* Can be perfectly usable
* Depends on depth of function call stack
* Depends on complexity of program
* If deep/large, then can become unweildy


### Error Handling C++ Style

{% code snippets/errorHandlingInCPP2.cc %}


### Outcome

* Code that throws does not worry about the catcher
* Exceptions are classes, can carry data
* Exceptions can form class hierarchy


### Consequences

* More suited to larger libraries of re-usable functions
* Many different catchers, all implementing different error handling
* Lends itself to layered software (draw diagram)
* Generally scales better, more flexible


### Practical Tips For Exception Handling

* Decide on error handling strategy at start
* Use it consistently
* Create your own base class exception
* Derive all your exceptions from that base class
* Stick to a few obvious classes, not one class for every single error


### More Practical Tips For Exception Handling

* Look at [C++ standard classes](http://www.cplusplus.com/reference/exception/) and [tutorial](http://www.cplusplus.com/doc/tutorial/exceptions/)
* An exception macro may be useful, e.g. [mitk::Exception](https://github.com/MITK/MITK/blob/master/Modules/Core/include/mitkException.h) and [mithThrow()](https://github.com/MITK/MITK/blob/master/Modules/Core/include/mitkExceptionMacro.h)
* Beware side-effects
    * Perform validation before updating any member variables

{% endidio %}
