---
title: Essential Reading
---

## Essential Reading

### General Advice

* Do a version control course such as [Software Carpentry](https://www.ucl.ac.uk/isd/services/research-it/training/courses/software-carpentry-workshop), or [MPHYG001](http://github-pages.ucl.ac.uk/rsd-engineeringcourse/).
* Contribute to online open-source project
* For your repository - pick a coding style, and stick to it


### Daily Reading

* Every C++ developer should keep repeatedly reading at least:
    * [Effective C++][Meyers], Meyers
    * [More Effective C++][Meyers], Meyers
    * [Effective STL][Meyers], Meyers
    

### Additional Reading

* Recommended
    * [Accelerated C++](https://www.amazon.co.uk/Accelerated-Practical-Programming-Example-Depth/dp/020170353X/ref=sr_1_5?ie=UTF8&qid=1484566101&sr=8-5&keywords=Moo+C%2B%2B), Koenig, Moo.
    * [Design Patterns (1994)](https://www.amazon.co.uk/Design-patterns-elements-reusable-object-oriented-x/dp/0201633612/ref=sr_1_1?ie=UTF8&qid=1484566062&sr=8-1&keywords=Design+Patterns), Gamma, Help, Johnson and Vlassides
    * [Modern C++ Design](https://www.amazon.co.uk/Modern-Design-Generic-Programming-Patterns/dp/0201704315/ref=sr_1_2?ie=UTF8&qid=1484566008&sr=8-2&keywords=Modern+C%2B%2B), Andrei Alexandrescu

    
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

[Meyers]: http://www.aristeia.com/books.html

