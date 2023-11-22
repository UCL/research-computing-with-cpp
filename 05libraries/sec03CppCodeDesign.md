---
title: C++ Code Design Summary
---

# C++ Code Design Summary

This is the end of our coverage of basic C++ language features. In the coming weeks, we'll explore different programming strategies, tools such as debuggers and profilers, and writing performant code using optimisations and parallel programming. Now is a good time to reflect on some of the features that we've learned about and how they fit together.

## General C++ Principles 

- Separate function / class declarations and implementations into header (.h) and source (.cpp) files. 
- Use smart pointers for data-owning pointers: manual memory management should be minimised. 
- Check for standard implementations of functions before writing your own: things like sorting are already well covered!
- The standard library offers performant containers such as `vector`, `array`, and `map`. 
- Make use of modern C++ features like range based loops, `auto` type inference, and anonymous functions where they make your code easier to understand or more flexible.
    - Be aware of possible performance issues with anonymous functions / `std::function` due to calling overheads.
    - Don't use `auto` if it makes it difficult for people to understand what types you are using.
- Don't import entire large namespaces like `std` as they risk name clashes.  
- Code should be modularised:
    - Functions should achieve a single task.
    - Classes should bundle together data and functionality necessary to represent a single concept.
    - Use unit-testing to test individual pieces of your program independently.
    - If you start repeating yourself in your code, try to refactor so that repeated segments are replaced with function calls. 
- Make use of features like intefaces and templates for flexible and reusable code.
- Programming solutions are not one size fits all: think carefully about your problem, the use case that you are developing for, and how you feel you can best serve your priorities and reflect the logical structure of your model in C++. 

## Run-time and Compile-time Polymorphism

Now that we've met both inheritance based run-time polymorphism and generic programming through templates, it's worth looking at the similarities and differences between the two. 
- Polymorphism allows for differences in behaviour to be decided at run-time. 
    - Behavioural differences are encoded inside classes which are related by inheritance. 
    - A single polymorphic function can operate on a base type and all its derived types. This is usually achieved by passing a pointer to the base type and calling virtual functions. 
- Templates (and function overloading) allow for differences in behaviour to be decided at compile-time. 
    - Behavioural differences are encoded into the external functions (or classes) which make use of the templated or overloaded type. The types which can be used aren't generally related by inheritance, but merely need to fulfil the functionality demanded in the templated code. 
    - Templates generate separate classes / functions for every different template parameter it is called with. 

There is a difference between a code which needs to have different behaviour with different objects not knowing ahead of time the exact type of that object (run-time polymorphism) and code which can be applied to applied to different types in different parts of the program, but does not require those types to be substitutable at run-time (compile-time polymorphism e.g. templates and function overloading). For example, you may well use the (overloaded) `+` operator to add integers together, and to concatenate strings, but you are unlikely to process data which could be _either_ an int or a string without knowing which it will be.

## Composition and Inheritance

We've seen this week that we can use multiple inheritance to implement multiple interfaces, which can lead to difficulties like the diamond problem, as well as making our model increasingly complex. While we can in fact inherit from an arbitrary number of base classes, it risks collisions between class namespaces and general confusion over the purpose and nature of an object. Multiple inheritance should only be used when motivated by genuine substitution (an "is-a" relationships, one class is a sub-type of the other) and a meaningful polymorphic use case. If faced with a multiple inheritance use case, consider whether it should in fact be represented as a chain of inheritance, or whether the functionality should in fact be refactored into a composition instead. 

Inheritance is sometimes misused by C++ programmers to share functionality between classes where composition would be clearer and more effective. Composition representing functionality is particularly powerful when combined with templates as we can still write a single piece of code which can be re-used with many types.

- Classes with overlapping functionality don't necessarily need to be related by some base class. 
- Classes should only be related by inheritance **if these classes should be interchangeable at some level** (i.e. can be substituted into the same place) in your code. For example, if we need a container such as a `vector` to be able to store and iterate over a diverse set of objects which are related by a core set of properties defined in a base class. 
- Mere sharing of functionality can often be better represented by wrapping said functionality in a class and including it in your other classes by composition. 
    - For example many classes will need to store data in a container such as a `vector`, but that does not mean they should inherit from the container class! They should have an instance of that container where they can store their data. 
- Multiple inheritance is generally limited to implementing two distinct, usually abstract, interfaces. An example of multiple inheritance in the C++ standard library is `iostream` (input/output stream) inherits from `istream` (input stream) and `ostream` (output stream). (See [`iostream` documentation](https://cplusplus.com/reference/istream/iostream/) and [C++ core guidelines on multiple inheritance](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rh-mi-interface).) 
- Inheritance is good at defining what a class _is_, but you can use composition for things that your class _makes use of_. 

## Useful References

### C++ Core Guidelines

There are many differing opinions about what exactly constitutes "good practice" in C++, but a good place to start looking is generally the [C++ core guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines). 

These guidelines are co-written by the original designer of C++ and are quite extensive, but you can select individual topics to explore when you are unsure of things. 

The following books are also useful, and available through UCL library services.

### A Tour of C++

The book [A Tour of C++](https://www.stroustrup.com/tour2.html) by Bjarne Stroustrup (one of the designers of C++) is a good, practical introduction to the major features of C++. The second edition is up to C++17, and the third edition covers up to C++20. You can check towards the back of the book what features become available in which C++ standard, so you can make sure you stay compatible with your compiler! 

### Effective Modern C++

The book [Effective Modern C++](https://www.oreilly.com/library/view/effective-modern-c/9781491908419/) is a good introduction to C++ up to the C++14 standard, and may be of help if you want to spend more time working on your C++ fundamentals. (Most, but by no means all, of the features that we have covered in this course are present in C++14.)

### Design Patterns

The book [Design Patterns](https://www.oreilly.com/library/view/design-patterns-elements/0201633612/) provides many examples of frequently occuring design solutions in object oriented programming that we have not covered in these notes. If you're comfortable with the ideas we've covered in C++ and want to improve your object-oriented software engineering skills, this book may be helpful. 