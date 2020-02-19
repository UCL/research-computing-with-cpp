---
title: Error Handling 
---

{% idio cpp %}

## Error Handling

### Exceptions 

* Exceptions are the C++ or Object Oriented way of Error Handling

### Exception Handling Example

{% code snippets/errorHandlingInCPP.cc %}


### What's the Point?

* A good summary [here](https://msdn.microsoft.com/en-us/library/hh279678.aspx): 
    * Have separated error handling logic from application logic
    * Forces calling code to recognize an error condition and handle it
    * Stack-unwinding destroys all objects in scope according to well-defined rules
    * A clean separation between code that detects error and code that handles error

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


### Comments

* Code that throws does not worry about the catcher
* Exceptions are classes, can carry data
* More suited to larger libraries of re-usable functions
* Many different catchers, all implementing different error handling
* Generally scales better, more flexible


### Practical Tips For Exception Handling

* Decide on error handling strategy at start
* Use it consistently
* Create your own base class exception
* Derive all your exceptions from that base class
* Stick to a few obvious classes, not one class for every single error

### Homework 17

* Taking the `Fraction` class from homework 8: 
    * Try to call  `simplify`  for a fraction with a denominator of 0 and see what exception is thrown
        * Try to catch this exception from the calling code
    * Create your own exception class that is thrown instead. It should inherit from `std::exception` (see [cppreference/error/exception](https://en.cppreference.com/w/cpp/error/exception)) 
        * Catch this from the calling code
    * Create the fraction and call `simplify`  from within a function that is called from the main calling code
        * Check you can catch the exception either in the calling code or from within the function 
    
### More Practical Tips For Exception Handling

* Look at [C++ standard classes](http://www.cplusplus.com/reference/exception/) and [tutorial](http://www.cplusplus.com/doc/tutorial/exceptions/)
* An exception macro may be useful, e.g. [mitk::Exception](https://github.com/MITK/MITK/blob/master/Modules/Core/include/mitkException.h) and [mithThrow()](https://github.com/MITK/MITK/blob/master/Modules/Core/include/mitkExceptionMacro.h)

{% endidio %}
