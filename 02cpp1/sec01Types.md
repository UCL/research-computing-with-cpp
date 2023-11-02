---
title: Types
---

# Types in C++

In this section we will cover some core ideas that you will need to understand to program effectively in C++ and similar languages. We will assume that you have some programming experience although not necessarily in C++, with the expectation that Python is the most commonly known language. As a result, a few things may need to be explained before proceeding with writing C++ code. 

## Type Systems

Type systems are an enormous topic, but we should understand a little bit about types in order to know how to program in different languages. Almost every high-level programming language is typed in some way. 

- Types denote what kind of data a variable represents. All data in a computer is just `1`s and `0`s, so we need to know how to _interpret_ data in order to use it. Types can make this process easier, since we don't have to manually remember what each memory location is supposed to be representing and enforce that it is treated appropriately. 
    - Consider for example how to print some data to the screen. The same sequence of bits at a memory location has a very different meaning if it's an integer or a series of characters. 
- An important property in many languages is **type safety**. This is a property of programs that essentially tells us that we are never using data meant to represent one type in a place where data of another type is expected. For example, we are never passing a string to a function which is meant to manipulate integer. This eliminates a large class of bugs!
- Programming languages may be statically or dynamically typed. 
    - In a **statically typed** language the type of a variable is decided at the declaration and it cannot be changed. To change a variable from one type to another requires another variable to be declared and an appropriate conversion to be defined. This is how typing works in C++, C, Rust, and many others. 
    - In a **dynamically typed** language the type of a variable can change throughout the program run-time. (Remember that a _variable_ is a handle, generally for some _data_ which resides in memory.) Instead of applying a type to the variable, the data itself is tagged with a type. A variable can have its data changed, including its type, for example from an `int` to a `string`. This is how typing works in a language like Python, Javascript, or Julia. 

### C++ Types

Types in C++ are static, and type correctness is checked at _compile time_. This means that if your program compiles correctly, you will not encounter type errors at runtime.**

Although dynamically typed programming languages like Python will prevent poorly typed statements from executing at runtime, there is no way to know whether your program contains type errors until you crash into one. Part of the problem is that functions can return different types depending the input or program state, meaning that you can't necessarily be sure that the thing that you think is calculating an integer is definitely going to give you an integer every time. Variables may also have their type changed by side effects after being passed to a function in a dynamically typed language. 
- These problems are not uncommon: conversion of a variable from an integer type to a floating point type under some circumstances is easy to do in Python. Floating point and integer types are interchangeable in most Python code but behave differently (integer arithmetic is exact while floating point is approximate, for example) and you may not discover the conversion has happened until you try to use the variable somewhere that a float cannot be used, such as indexing an array. Because the conversion is silent and valid in a dynamically typed language, it can be extremely hard to find _where_ the conversion happened in a large program, as the problem could originate a long way away from where the type error gets raised! 

Type systems can be leveraged to ensure many kinds of safety properties in programs because information can be built into custom types. Examples of this might be ensuring at compile time that matrices in matrix multiplications are compatible (recall that a $X \times Y$ matrix can be multiplied by a $Y \times Z$ matrix), or that physical dimensions are consistent (e.g. a velocity has units $\text{Length} \times \text{Time}^{-1}$, so $v = \frac{d}{t}$ is a valid expression but $v = \frac{d^2}{t^2}$ is not). 


> \* A low-level programming language is a language that closely resembles the instruction set of the machine itself (we'll see more about this idea in week 7). Basically your machine can do a fixed set of simple operations, and low-level languages deal with these directly. High-level languages are more abstract, and they allow for more complex operations in individual instructions. Even declaring a variable is a high level concept! High-level languages are converted to machine operations by compilers, and are hardware independent as long as there is a compatible compiler which can translate the high-level code into that machine's instruction set. High-level languages specify what the program should do, but not _exactly_ how it should actually be done, so different compilers should result in programs which produce the same results but can actually have a different sequence of operations. 

> \** Technically C++, like a number of other languages, has some features which are not type-safe. It is possible in C++ to subvert the type system by using some low level memory operations, but there is almost never a reason to do this so you're unlikely to see this in practice and you shouldn't do it in your own code. Attempting to do this kind of manipulation usually results in **undefined behaviour**, so you won't necessarily even be able to predict what your program will do unless you know exactly how your compiler turns your source into machine code!

## C++ Types and Declaring Variables

In C++ when declaring variables we do so by first declaring the type, then the name of the variable, and then its value. For example:

``` cpp
int x = 5;
```

- This declares a variable `x` of type `int` with value `5`. 
- Some types have default initial values, which would mean that you don't have to supply the value explicitly. 
- Some types can be declared _unitialised_, which means that the memory for that variable is reserved but not initialised. It will contain whatever bits were already there! It's a good idea to initialise variables explicitly.

Types in C++ can sometimes be verbose or complicated, and it is sometimes easier to read and write code which makes use of the `auto` keyword. This keyword tells the C++ compiler to deduce the type for us. This is called _type inference_ and was made a feature of C++ in C++11, so will be absent in older codes.

``` cpp
auto x = 5;
auto y = std::abs(-13.4);
```

- `auto` can usually deduce the type from the result of an expression by looking at the types of all the variables and functions within it.
    - For example here it interprets `5` as an integer and therefore deduces that `x` is an int. 
    - It will deduce that `y` is a `double`, since `-13.4` must be a floating point type (`double` by default) and `std::abs` returns a `double` when given a `double`.
    - Be especially careful when values can be ambiguous. Here `5` is being assigned as an int, but `5` is also a valid `float` or `double`. **If you want a specific type in cases like this you should always specify it explicitly**.
- `auto` doesn't always work: the compiler must be able to deduce the type from contextual information.
    - You cannot declare an unitialised variable with `auto` e.g. `auto z;` will lead to a compiler error as it won't know what type is intended for `z`, even if `z` is later assigned. 
- You cannot use `auto` when declaring the return types or parameter types of functions, you must always declare these type explicitly. 
    - It's generally a good idea therefore to know what the types of variables in your code are, even if you choose to use the `auto` keyword! This will make writing your own functions, and knowing what functions you can pass your data to, much easier. 
- In an IDE like VSCode you can inspect the type of a variable by hovering the mouse over that variable. If you've used `auto` it will be able to tell you what type has been deduced for it. 
- Bear in mind that `auto` can make your code more concise, but can also make your code harder to understand. Sometimes it's better to write your type explicitly so that people reading your code can immediately understand what the types of your variables are. 

## Defining Custom Types

Custom types are an important feature in typed languages in order to be able to represent and manipulate more complex data in a type-safe way. In C++ the most common way to define a new type is to declare a `class` (or equivalently a `struct`). Classes are a common feature of **Object Oriented Programming**, which is a popular approach to programming in C++. (Some examples of other languages with classes for object oriented programming are C#, Java, and Python.) We'll discuss the design and use of classes in the next section, so for now let the following suffice:
- A class is a custom data type which is defined by the programmer. It can contain any number of variables and functions.
- Once it is defined it can be used like any other type, e.g. it can be accepted as an argument in, or returned from, a function. Type safety rules still apply. 
- Classes give us a way of defining sub-types which are _substitutable_. For example we can define a `Shape` type, and then have `Circle` and `Square` sub-types which are accepted by the type system anywhere where a `Shape` type is accepted. This makes our type system more flexible and expressive. We discuss classes in detail in a later section of this week's notes. 

We will focus overwhelmingly on classes as our means of defining custom types, but for those who are interested there are two further ways of declaring custom types in C++:
- `enum`: This stands for _enumeration_. An `enum` is a type which can take one of a finite set of values (i.e. the values are _enumerable_). Each of these values must have a name, for example let's say we want a `Colour` enum which can take the values `red`, `green`, and `blue`. We can declare a new `enum` called `Colour` in two ways:
    - `enum Colour {red, green, blue};`. This kind of enum implicitly converts the values `red`, `green`, and `blue`, to `1`, `2`, and `3` respectively, and the `Colour` type can be used interchangeably with `int`. 
        - Because this type of `enum` is interchangeable with `int`, it can be used to e.g. index an array. This can be useful when you want to efficiently store data based on categorisations. For example, say you have data about some population, split up by gender and age group. By turning your gender categories and age groups into enums, you can then store your data as a matrix which is indexed like `data[gender][age_group]`. 
        - For this kind of enum we can just reference these values using the names `red`, `green`, and `blue`.
    - `enum class Colour {red, green, blue};`. This kind of enum (called an `enum class`) cannot be used interchangeably with `int`, and therefore `Colour` can only be used in places that are explicitly expecting a `Colour` type. **We usually want to use an `enum class` so that we don't accidentally mix it up with integer types!**
        - This cannot be used to index arrays (because it is not an int), but it can be used as a key in `map` types. `map` and `unordered_map` provide C++ equivalents to Python's dictionary type. 
        - In order to use these values we have to also include the class name, so we have to write `Colour::red`, `Colour::green`, or `Colour::blue`. 
- `union`: Union types are types which represent a value which is one of a finite set of types. A `union` is declared with a list of members of different types, for example `union IntOrString { int i; string s; };` can store an `int` or a `string`. When a variable of type `IntOrString` is declared, it is only allocated enough memory to store _one_ of its members at a time, so it cannot store both `i` and `s` at the same time. The programmer needs to manually keep track of which type is present, often using an auxilliary variable, in order to safely use union types. Given this additional difficulty, **I wouldn't recommend using union types without a very strong reason.**

Microsoft has excellent, and accessible, resources on [`enum`](https://learn.microsoft.com/en-us/cpp/cpp/enumerations-cpp?view=msvc-170) and [`union`](https://learn.microsoft.com/en-us/cpp/cpp/unions?view=msvc-170) types if you are interested in learning more about them. 

**N.B.** C++17 onwards also has a special class called `std::variant` which is designed to replace `union` types in a more type-safe way, because the `variant` can be checked to see which type it is currently holding. (That said, checking which type the variant has is still rather clunky, and you have to check for each type manually so if there are many cases it can be easy to miss one and the compiler will not warn you!) Ultimately, union / variant types are not terribly common in C++ code in practice, although some languages (especially functional languages like ML and Haskell) handle these concepts much more naturally. If you're interested in this kind of approach to types, I recommend reading up on [algebraic datatypes](https://en.wikipedia.org/wiki/Algebraic_data_type).
