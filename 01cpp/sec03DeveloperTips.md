---
title: Developer Tips
---

## Developer Tips

### Practical Tips

(This section is more of an open-ended discussion to gauge
where the class is at in terms of experience.)

* If you feel like:
    * With more coding, more things go wrong
    * Everything gets messy
    * Feeling that you're digging a hole
* Then we provide:
    * Pragmatic tips as how to do this in practice
    * Adapted for scientific research


### Coding tips

* Follow coding conventions for your project 
* Compile often
* Version control
    * Commit often
    * Useful commit messages
        * don't state what can be diff'd
        * explain why
    * Short running branches
    * Covered on [MPHYG001][MPHYG001]    
* Class: "does exactly what it says on the tin"
* Class: "build once, build properly", so testing is key.


### C++ tips

Numbers in brackets refer to Scott Meyers "Effective C++" book.

These are some of my favourites.

* Declare data members private (22)
* Use `const` whenever possible (3) 
* Make interfaces easy to use correctly and hard to use incorrectly (18)
* Avoid returning "handles" to object internals (28) 
* Initialise objects properly. Throw exceptions from constructors. Fail early. (4)
* Never throw exceptions from destructors
* Prefer non-member non-friend functions to member functions (better encapsulation) (23) 


### OO tips

* Make sure public inheritance really models "is-a" (32) 
* Learn alternatives to polymorphism (Template Method, Strategy) (35) 
* Model "has-a" through composition (38) 
* Understand [Dependency Injection][DependencyInjection]
* i.e. most people overuse inheritance


### Scientific Computing tips

* Papers require numerical results, graphs, figures, concepts
* Optimise late
    * Correctly identify tools to use
    * Implement your algorithm of choice
    * Provide flexible design, so you can adapt it and manage it
    * Only optimise the bits that are slowing down the production of interesting results
* So, this course will provide you with an array of tools


### Further Reading

* Every C++ developer should keep repeatedly reading at least:
    * [Effective C++][Meyers], Meyers
    * [More Effective C++][Meyers], Meyers
    * [Effective STL][Meyers], Meyers
    * Design Patterns (1994), Gamma, Help, Johnson and Vlassides
    
[Meyers]: http://www.aristeia.com/books.html
[MPHYG001]: https://moodle.ucl.ac.uk/course/view.php?id=28759
[DependencyInjection]: http://en.wikipedia.org/wiki/Dependency_injection
