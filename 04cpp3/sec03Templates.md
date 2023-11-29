---
title: Templates
---

Estimated Reading Time: 45 Minutes

# Templates

Templates are a way of writing generic code which can be re-used with different types. This is similar to the polymorphism that we have seen previously through class inheritance, except that the typing for a template happens at compile time rather than runtime. 

Templates in C++ come in two main kinds:

- Function Templates
- Class Templates

When a class or function template is used to instantiate an concrete class or function using a specific type, a new class or function definition is created for each type with which the template is instantiated. So unlike our inheritance based run-time polymorphism, where we have one function which can take multiple classes with overlapping definitions (defined by inheritance from a base class), now we use one piece of code to generate multiple separate functions (or classes), each of which accepts a different type. This is why the compiler must know all the types which are to be used in templated functions at compile time. This is sometimes known as "static polymorphism". 

## Using Templates with Classes

You have already been using templates in this course: `vector<>` is an example of a _class template_. The `vector` class template is defined the type of the elements of the vector left as a template argument: you provide this argument in the angle brackets (`<>`) when you declare your vector object. This then tells the compiler what kind of data your vector will hold, so that it can properly handle it (allocating memory, iterating through elements, type checking for elements which are used in the code etc.). 

**N.B.** A note on terminology:
- A **class template** is not a class, it is a template which can be used to generate the definition of a class once the template arguments have been filled. For example, `vector<>` is a class template which can be used to generate vector classes which hold different types of data, but you cannot create a vector object with no type in the angle brackets. 
- A **template class** is a class that has been created from a template. For example `vector<int>` is a template class that has been generated from the `vector` class template by providing the type `int` as a template argument. `vector<int>` is an entirely separate class from `vector<double>`.

The other containers in the standard library are also class templates, which require you to provide template arguments to instantiate concrete classes. Note that some may take more than one template argument, for example `map<>` takes two types: one for the keys and one for the values. Type for a map from strings to integers would be `map<string, int>`. 

We can define a class template using the following syntax:
```cpp
template<typename T>
class myClassTemplate
{
    public:
    myClassTemplate(T value)
    {
        myMemberT = value;
    }

    private:
    T myMemberT;
};
```
- `T` is the template parameter, and the `typename` keyword tells us that `T` must denote a type. (You can equivalently use the `class` keyword.)
    - Do note that you don't need to call your template parameter `T`; like function parameters or other variables, it can have any name. It's good to give it a more meaningful name if the type should represent something in particular, for example `matrixType` could be the name if your templated code deals with arbitrary types representing matrices. This is especially useful when using templates with multiple template parameters! 
- We can then use `T` like any other type inside the body of the class definition. 
- Additional template parameters can appear in the angle brackets in a comma separated list e.g. `template<typename T1, typename T2>`. This is how e.g. `std::map` works. 

**Template parameters do not have to be `typename`, i.e. we are not limited to simply templating types.** You can also have template parameters that are values such as an `int`, or a `bool`, or any other type. These can be used to define special versions of classes with separate implementations when provided with particular values. For example we might have `template<int maxSize>` to define different classes depending on the maximum size of data it will accept.
- `std::array` is a good example of a class template which take both a type and a value: the type of the elements and the number of elements in the array.
- Having values as template parameters means that they must be constants known at compile time. 
- Using template parameters which are values can allow you to leverage to type system to enforce correctness on your program. For example, if your program models objects in 3D space, then you will need a representation of a 3-vector. If you use `std::vector<double>` then these vectors could be any size, so you have to make sure manually that no vectors of other sizes can sneak into your program. If you use `std::array<double, 3>` to represent a 3-vector then the compile will enforce that all positions, velocities, and so on are 3 dimensional. (If you work in general relativity, then this can also help you define different types for 3-vectors (`std::array<double, 3>`) and 4-vectors (`std::array<double, 4>`)!)

**N.B.** Templates which have many parameters (types or values) can make type names quite long, so if there is something that you want to use frequently you may consider giving it an alias using the `using` syntax:
```cpp
using Vec3 = std::array<double, 3>;
```
This can also make your type names more meaningful to people reading your code. 

## Template Classes and Inheritance 

Consider a class template which takes one type as a template argument, such as `vector`. 
Let us say we have two classes `A` and `B`, where `B` is a sub-class of `A`. It is important to understand that the template classes `vector<A>` and `vector<B>` **do not share the same relationship** as `A` and `B` i.e. `vector<B>` is not a sub-class of (does not inherit from) `vector<A>`. 

## Function Templates

As well as creating templates for classes we can also create templates for functions when we want to use the same code to describe the behaviour of a function taking a variety of different types. In fact, most class template definitions will also contain function templates, since member functions are likely to be dependent on template parameters. 

Function templates can have the same kinds of template parameters as class templates. 

The syntax for declaring function templates is essentially identical to declaring a class template:
```cpp
template<typename T>
T templatedAdder(T a, T b)
{
    T c = a + b;
    return c;
}
```
- This function can only be created for a given type if the function body forms valid expressions when the type is substituted for `T`.
    - In this case the restriction is that the operators `=` and `+` must be defined for type `T`. 
    - This applies to integers, double, strings etc. (see below on Operator Overloading for more information). 

A common example of a templated function would be a function which acts on a container type but doesn't need to access the data itself. Consider this example which take every other element of a vector:
```cpp
template<typename T>
vector<T> everyOther(vector<T> &v_in)
{
    vector<int> v_out;
    for(size_t i = 0; i < v_in.size(); i++)
    {
        if(i % 2 == 0) v_out.push_back(v_in[i]);
    }
}
```
- The exact details of the type `T` don't matter in this case, since we never access the data of type `T` anyway. The only restriction on `T` is that it can be added to a vector.
- A function can be generated for every kind of vector in this way. 

## Using Templates with Overloaded Functions 

One very useful way to make use of templates is to exploit operator / function overloading. Operators or functions which are "overloaded" can operate on multiple types, for example:
- The arithmetic operators `+`, `-`, `*`, and `/` are defined for a variety of types including `int`, `float`, and `double`. It's often easy to write numerical functions which operate on generic numerical types using templates. The `templatedAdder` example above will work on any C++ type for which `+` is well defined. 
- The `[]` operator can be used to access many kinds of data-structures including `vector`, `array`, `map`, and C-style arrays. We can therefore write functions which are agnostic about the precise kind of storage used, as long as the same code will work for all of the types that we are interested in.
- The pointer dereferencing operator `*` works on unique, shared, and raw pointers, so we can write template code which can be used with any of these types if we don't use functionality that is specific to one or the other of them. 

When we are designing classes we can overload operators ourselves. This can be very important when defining types that we want to be able to use within templated functions. Consider for example a fraction type:
```cpp
class Fraction
{
    public:
    Fraction(int a, int b) : numerator(a), denominator(b) {}

    private:
    int numerator;
    int denominator;
};
```
- This class represents a rational number: a ratio of two integers. 
- This is an appropriate arithmetic type, so it would make sense for us to define operators like `+`, `-`, `*`, and `/` for the Fraction type. 
    - This would allow us to pass it to templated functions that deal with generic arithmetic types. 
    - This can therefore be more flexible and intuitive than defining member functions for these kinds of operations. 

Let's define the `*` (multiplication) operator. (The others can be defined similarly.) We can do this in one of two ways. 

A) As a member function:

```cpp
class Fraction
{
    public:
    Fraction(int a, int b) : numerator(a), denominator(b) {}

    Fraction operator* (Fraction y)
    {
        return Fraction(numerator * y.numerator, denominator * y.denominator);
    }

    private:
    int numerator;
    int denominator;
};
```

- The member function `operator*` _can_ be called like any other member function.
    - If you have two `Fraction` objects `f1` and `f2` you could calculate a new fraction `Fraction f3 = f1.operator*(f2);`. 
- However this function also overloads the `*` infix operator.
    - So we can now write `Fraction f3 = f1 * f2;`.
- Note that because it is a member function, it has access to the private member variables `numerator` and `denominator`. 

B) Outside the class

```cpp
class Fraction
{
    public:
    Fraction(int a, int b) : numerator(a), denominator(b) {}

    int getNumerator(){ return numerator; }
    int getDenominator(){ return denominator; }

    private:
    int numerator;
    int denominator;
};

Fraction operator*(Fraction x, Fraction y)
{
    int numerator = x.getNumerator() * y.getNumerator();
    int denominator = x.getDenominator() * y.getDenominator();
    return Fraction(numerator, denominator);
}
```

- This version also overloads the `*` infix operator in the same way. 
- There is now no member version so we can't call `f1.operator*(f2)`, but we can call the non-member function `operator*(f1, f2)`. 
- Note that we had to create `get` functions for the private member variables because `operator*` is not a member function in this case and so does not have access to private members. 

## Using Class Members with Templated Types

Writing a templated function which takes type `T` implicitly defines an interface that must be met by `T`. 

- In the simple adding example we use the `+` operator on objects of type `T`: therefore `T` must implement this operator in order for the function to be successfully generated. 
- In our vector sampling example we only require that `vector<T>` is a valid type; this places very few restrictions on `T` (although there are some). 

The only restrictions on our templates is that the code is valid once the type substitution has been made. If we have a template parameter `T` then we can access member variables and functions of the class `T`, and this function will be able to be generated for any class which implements those properties. For example, let's take this function:

```cpp
template<typename T>
T& getTheBiggerOne(T &a, T &b)
{
    if(a.getArea() >= b.getArea())
    {
        return a;
    }
    else
    {
        return b;
    }
}
```

- This function will return whichever of its two inputs has the larger area.
- This is a valid template for any type which implements the member function `getArea()`.
- Note we don't specify the return type of `getArea()`; for this code to be valid it is only necessary that `>=` is defined for the return type of `getArea()`.

We can call this function using our `Shape` classes (`Shape`, `Circle`, `Square`) from last week, since they implement `getArea()`. But we could also implement a new class, like this one:

```cpp
class Country
{
    Country(string n, double a, double p) : name(n), area(a), population(p) {}

    double getName() { return name; }
    double getArea() { return area; }
    double getPopulation() { return population; }

    private:
    string name;
    double area;
    int population;
}
```

- This class defines countries by their name, area, and population. 
- This class also fulfills all the conditions of `getTheBiggerOne`:
    - `getArea()` is implemented.
    - `>=` is defined for `double`. 

We can use `getTheBiggerOne` with our `Country` class just as well as our `Shape` class, even though they are not (and certainly should not be) related by any inheritance! Templates allow us to define generic code that is broader than inheritance based polymorphism, on the condition that the type can be determined at compile time in order to generate a statically typed function.  

- Class inheritance provides run-time polymorphism: I can define one function that takes a base class (e.g. `Shape`) and objects of that class or its derived classes can be passed to it. The compiler does not need to know at compile time whether the function will end up receiving `Shape` or `Circle` or `Square`. 
- Templates provide static polymorphism. I can define one function template that generates separate functions for each class. If I want to use my function with both `Shape` and `Country`, the compiler needs to know this at run time.
    - I can't declare a single function or class (such as a container), which can take both `Shape` and `Country`. For example, I can't put a `Shape` object in the same vector as a `Country` object, since it either needs to be a `vector<Shape>` or `vector<Country>`. 
    - If I use the function with `Shape` and with `Country` in the same program, I will actually generate two functions: `Shape& getTheBiggerOne(Shape&, Shape&)` and `Country& getTheBiggerOne(Country&, Country&)`. These functions are separate because they have different signatures (parameter and return types). 
- These two can be combined. For example, `getTheBiggerOne` is a template which could be instantiated with the type `Shape`. The resulting function, which takes and returns references to `Shape`, could be used with objects of type `Shape`, `Circle` or `Square` (run time polymorphism based on their inheritance tree) but not `Country` (this is not part of the same inheritance tree). 

## Organising and Compiling Code with Templates

Since templates create concrete classes and functions at compile time, the compiler needs to have access to the template definition and the argument(s) for which it needs to be instantiated together at compile time. 

Well organised C++ code typically organises code into declarations and implementations:

- Declarations are contained in header files, usually ending in `.h` or `.hpp`. (Sometimes `.h` is used to distinguish header files intended for use with C and `.hpp` for header files which are only compatible with C++, but this is just a convention and not uniformly applied.) These are typically there to declare classes, and the variables and functions that they possess, and/or the variables and functions contained in a particular namespace. Function declarations just contain the name and signature (input types and return type) of the function, but not the actual code that it executes. When writing code which uses a class we usually just include the header file in our code to include the declaration: as long as we know the interface for the class we can compile code down to an object file which interfaces with that class. The final code can be created by _linking_ with the class object file which defines the actual implementation that needs to be executed. 
- Implementations are contained in source files, usually ending in `.cpp` for C++ files. These contain the actual code which is executed by the functions declared in the header. (There may also be functions declared in the source file which aren't in the header if they're only locally needed and therefore don't need to be included elsewhere.) This can then be compiled down to an object file for linking with other object files which interface with each other. 

We have to be a little careful when working with this model in the case of templates. We'll explore this using a function template, although the considerations are the same for a class template as well. Let's declare a function in a header file, and write two source files: one which implements the function in the header, and one which makes use of this function. 

First the declaration in `declaration.hpp`:

```cpp
#ifndef DECLARATION_HPP 
#define DECLARATION_HPP

namespace utilFunctions
{
    int add(int, int);
}

#endif
```

- It's usually a good idea to put functions and variables which are defined in a header but not part of a class inside a namespace, to avoid potential name clashes. 
- The `#ifndef`, `#define` and `#endif` are pre-processor directives. This pattern is called an "include guard": it means that the file's contents will be ignored if it has already been included somewhere else in the same compilation unit, so that the contents are not declared twice (which would cause an error). 
    - An alternative to these include guards which you will have seen already is `#pragma once`. This is a common pre-processor directive to only include a file once in a compilation unit, but it is not part of the ISO C++ standard and therefore may not be compatible with all platforms and compilers. 

Then the implementation in `implementation.hpp`:

```cpp
#include "declaration.hpp"

int utilFunctions::add(int a, int b)
{
    int c = a + b;
    return c;
}
```

- We need to include `declaration.hpp` in order to have access to the namespace and function declaration. 

Finally we have another file which will want to use this function, which we'll call `usage.cpp`. 

```cpp
#include "declaration.hpp"
#include <iostream>

int main()
{
    int x = 15;
    int y = 27;

    std::cout << utilFunctions::add(x, y) << std::endl;

    return 0;
}
```

- This also needs to include `declaration.hpp` in order to access the function declaration. 
- Note that it does **not** need the function implementation: it only needs to know the name of the function and what types it accepts and returns to be able to compile.

We can compile `implementation.cpp` and `usage.cpp` down to two separate object files and link them to produce a fully functional executable. 

This simple addition function could be made much more flexible by also operating on other numeric types, like `float` or `double` for which the addition operator is defined. In this case, rather than writing three separate functions out, all with essentially identical code, we could use a function template. 

Let's update our declaration to use a template:

```cpp
#ifndef DECLARATION_HPP 
#define DECLARATION_HPP

namespace utilFunctions
{
    template<typename T>
    T add(T, T);
}

#endif
```

and our implementation:

```cpp
#include "declaration.hpp"

template<typename T>
T utilFunctions::add(T a, T b)
{
    T c = a + b;
    return c;
}
```

If we try now to compile and link our executable we will find an error like this:

```bash
undefined reference to `int utilFunctions::add<int>(int, int)'
```

- The compiler has been unable to implement a definition of the `add` function for the type `int`, so this definition does not exist for us to use. 
- This error shows up during linking. You can compile both object files like before, because both match the template declaration and therefore are valid, but neither one can define the specific implementation that we want so when linking it finds that the function isn't defined anywhere. 
- `implementation.cpp` cannot define the implementation when compiled down to an object because it has the function template but not the intended type, so it can't come up with any concrete implementation. 
- `usage.cpp` cannot define the implementation when compiled down to an object because it knows what type it should be used for, but it doesn't have the templated implementation (this is in `implementation.cpp`, and we have only included `declaration.hpp`). 

There are two possible ways to approach this problem.
1. We can include the templated function implementation in the header file instead of a separate source file. 
    - In this case the compiler can use the template to create the function for whatever type is called for in `usage.cpp`.
    - Concrete function is only created from the template if it is actually used. 
    - This is flexible, but breaks the separation of declaration and implementation.
    - Can cause the size of the executable to increase because the definitions will be recreated in different compilation units. 

```cpp
#ifndef DECLARATION_HPP 
#define DECLARATION_HPP

namespace utilFunctions
{
    template<typename T>
    T add(T a, T b)
    {
        T c = a + b;
        return c;
    }
}

#endif
```

2. We can keep our header file with just the declaration, and tell the compiler which types to implement the function for in the source file (`implementation.cpp`). 
    - In this case, `usage.cpp` will only be able to use `add` for the types which are explicitly instantiated in `implementation.cpp`. 
    - This is less flexible as you need to anticipate any combination of template arguments that the function  will be used with, but keeps the declaration and the implementation separate.
    - Separate function implementations will be created for each set of types given, even if they are never used. 
    - It can also be useful if you want the function to restrict usage to a sub-set of possible types. 

```cpp
#include "declaration.hpp"

template<typename T>
T utilFunctions::add(T a, T b)
{
    T c = a + b;
    return c;
}

template int utilFunctions::add(int, int);
template float utilFunctions::add(float, float);
template double utilFunctions::add(double, double);
```

