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

## Undefined Behaviour

A quirk of the C++ programming language is that not all source code that compiles is actually a valid C++ program. **Undefined behaviour** refers to situtaions in C++ where the standard offers no guidance and a compiler can more or less do what it likes; as a result we as programmers may have little idea what will happen if such a program is run, and the results will vary from compiler to compiler, and system to system. This means if our program has undefined behaviour then even if we have thoroughly tested it on our own system, it may not be portable to anyone else's. 

You can read more about undefined behaviour on e.g. [cppreference](https://en.cppreference.com/w/cpp/language/ub). 

Much undefined behaviour centres around memory access or modification, for example:
- Reading values outside the bounds of an array.
- Reading / modifying an variable with a pointer of a different type (also known as type aliasing).
    - There is a special exception for the type `char` or `std::byte` which allow us to observe any variable / object data as a sequence of bytes.
    - You can read about [type aliasing here](https://en.cppreference.com/w/cpp/language/reinterpret_cast#Type_aliasing) which will also describe the concept of _similar types_ which we will not get into in these notes!
- Modifying a `const` value through a non-const pointer. 
- There are many more causes of undefined behaviour, but note that many involve doing something to invalidate some aspect of the program's definition: subverting the type system, undermining `const` declarations, accessing private members and so on. It's usually not the case that you will _accidentally_ cause undefined behaviour, but rather it is often because of attempts to use low-level access to memory to get around a high-level construct. 

Undefined behaviour is often the consequence of the meeting of C++'s lower level and higher level features in ways that are not valid. We will just give one simple example to illustrate why this kind of behaviour ends up being undefined: that of undermining `const`.

Consider the following code: what will it do?
```cpp
#include <iostream>

int main()
{
    const int N = 10;

    // Pointer ptr is a non const pointer to non const data
    // It is initialised to point to the same address as N is stored
    int *ptr = (int *) &N;
    *ptr += 1;

    std::cout << N << std::endl;
    std::cout << *ptr << std::endl;
    std::cout << ptr << " " << &N << std::endl;

    return 0;
}
```
The behaviour here is **undefined** since we have used an incompatible pointer type to read and modify the memory which contains the constant integer `N`. 

On my machine, the output is as follows:
```
10
11
0x7fffd87014cc 0x7fffd87014cc
```
- From the third line we can see that the storage address for `N` is the same as that pointed to by `ptr`.
- `N` is reported as `10`, but `*ptr` is `11`, which appears inconsistent! 
- This is because my compiler has been told that `N` is a constant, and so in the line `std::cout << N << std::endl` the (time expensive) memory read is replaced by a hard-coded value `10`, which is more efficient and the compiler will assume is valid _because we told it so_. 
- When printing out the value that the pointer is pointing to however, the memory read is necessary so we get the value `11`, which is the value which is actually stored in RAM. 
- If we were to use `N` again later in the program, what value we would get would simply depend on whether the compiler optimised out the data read or not!
- Other compilers may do different things under different circumstances - there are no guarantees!

Part of the price we pay for having this low level memory access is that it is possible to access memory in ways that violate the conditions that we have already stated: we can also set a pointer to look at any given location in memory (that our program has access to), which means it can be set to read or even modify `const` values, `private` members, variables of other types and so on. But in order for the compiler to do its best job, it needs to be able to make assumptions about the behaviour of the program and integrity of data, as we've seen with the above `const` violation example. 

- Make good use of things high level concepts like the type system, `const`, and access specifiers to make your program safer and more expressive. In almost all programming circumstances these things will allow the compiler to catch any violations of your model and prevent them from compiling.
- **Don't do daft things with low-level memory** to undermine that safety: in C++ _you have some responsibility to make use of the language properly_.
- Undefined behaviour can be hard to catch because compilers will not necessarily catch or even issue warning for undefined behaviour. (The above example for example will only issue a warning if compiled with the rather niche `-Werror=cast-qual` flag. Even the `-Wall`, "all warnings", and `-Wextra` flags will not be enough to catch this one!)
- Do learn about some of the causes of undefined behaviour. 

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