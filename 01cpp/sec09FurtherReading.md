---
title: Further Reading
---

## Further Reading

### General Advice

* Contribute to online open-source project
* Pick a coding style, and stick to it
* Do a version control course such as [Software Carpentry](https://www.ucl.ac.uk/isd/services/research-it/training/courses/software-carpentry-workshop), or [MPHYG001](http://development.rc.ucl.ac.uk/training/engineering/).


### C++ tips

Numbers in brackets refer to Scott Meyers "Effective C++" book.

* Declare data members private (22)
* Use `const` whenever possible (3) 
* Make interfaces easy to use correctly and hard to use incorrectly (18)
* Avoid returning "handles" to object internals (28) 
* Initialise objects properly. Throw exceptions from constructors. Fail early. (4)


### OO tips

* Never throw exceptions from destructors
* Prefer non-member non-friend functions to member functions (better encapsulation) (23)
* Make sure public inheritance really models "is-a" (32) 
* Learn alternatives to polymorphism (Template Method, Strategy) (35) 
* Model "has-a" through composition (38) 


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
