# Templates

Templates are a way of writing generic code which can be re-used with different types. This is similar to the polymorphism that we have seen previously through class inheritance, except that the typing for a template happens at compile time rather than runtime. 

Templates in C++ come in two main kinds:
- Function Templates
- Class Templates

When a class or function template is used to instantiate an concrete class or function using a specific type, a new class or function definition is created for each type with which the template is instantiated. So unlike our inheritance based polymorphism, where we have one function which takes multiple classes with overlapping definitions, now we use one piece of code two generate multiple separate functions (or classes), each of which accepts a different type. This is why the compiler must know all the types which are to be used in template instantiation at compile time. This is sometimes known as "static polymorphism". 

## Using Templates with Classes

You have already been using templates in this course: `vector<>` is an example of a _class template_. The `vector` class template is defined the type of the elements of the vector left as a template argument: you provide this argument in the angle brackets (`<>`) when you declare your vector object. This then tells the compiler what kind of data your vector will hold, so that it can properly handle it (allocating memory, iterating through elements, type checking for elements which are used in the code etc.). 

**N.B.** A note on terminology:
- A **class template** is not a class, it is a template which can be used to generate the definition of a class once the template arguments have been filled. For example, `vector<>` is a class template which can be used to generate vector classes which hold different types of data, but you cannot create a vector object with no type in the angle brackets. 
- A **template class** is a class that has been created from a template. For example `vector<int>` is a template class that has been generated from the `vector` class template by providing the type `int` as a template argument. `vector<int>` is an entirely separate class from `vector<double>`.

The other containers in the standard library are also class templates, which require you to provide template arguments to instantiate concrete classes. Note that some may take more than one template argument, for example `map<>` takes two types: one for the keys and one for the values. 

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
- We can then use `T` like any other type inside the body of the class definition. 
- Additional template parameters can appear in the angle brackets in a comma separated list e.g. `template<typename T1, typename T2>`.
- Template parameters do not have to be `typename`. You can also have a template parameter that is an `int`, or a `bool`, or any other type. These can be used to define special versions of classes with separate implementations when provided with particular values. For example we might have `template<int maxSize>` to define different classes depending on whether the maximum size of data it wlil accept. This kind of template parameter is much less common. 

## Template Classes and Inheritance 

Consider a class template which takes one type as a template argument, such as `vector`. 
Let us say we have two classes `A` and `B`, where `B` is a sub-class of `A`. It is important to understand that the template classes `vector<A>` and `vector<B>` **do not share the same relationship** as `A` and `B` i.e. `vector<B>` is not a sub-class of (does not inherit from) `vector<A>`. 

## Function Templates

As well as creating templates for classes we can also create templates for functions when we want to use the same code to describe the behaviour of a function taking a variety of different types. In fact, most class template definitions will also contain function templates, since member functions are likely to be dependent on template parameters. 

Function templates can have the same kinds of template parameters as class templates. 

The syntax for declaring fucntion templates is essentially identical to declaring a class template:
```cpp
template<typename T>
T templatedAdder(T a, T b)
{
    T c = a + b;
    return c;
}
```

## Using Templates with Overloaded Functions 

One very useful way to make use of templates is to exploit operator / function overloading. Operators or functions which are "overloaded" can operate on multiple types, for example:
- The arithmetic operators `+`, `-`, `*`, and `/` are defined for a variety of types including `int`, `float`, and `double`. It's often easy to write numerical functions which operate on generic numerical types using templates. The `templatedAdder` example above will work on any C++ type for which `+` is well defined. 
- The `[]` operator can be used to access many kinds of data-structures including `vector`, `array`, `map`, and C-style arrays. We can therefore write functions which are agnostic about the precise kind of storage used, as long as the same code will work for all of the types that we are interested in.
- The pointer dereferencing operator `*` works on unique, shared, and raw pointers, so we can write template code which can be used with any of these types if we don't use functionality that is specific to one or the other of them. 

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
```
undefined reference to `int utilFunctions::add<int>(int, int)'
```
- The compiler has been unable to implement a definition of the `add` function for the type `int`, so this definition does not exist for us to use. 
- This error shows up during linking. You can compile both object files like before, because both match the template declaration and therefore are valid, but neither one can define the specific implementation that we want so when linking it finds that the function isn't defined anywhere. 
- `implemenation.cpp` cannot define the implementation when compiled down to an object because it has the function template but not the intended type, so it can't come up with any concrete implementation. 
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
    - In this case, `usage.cpp` will only be able to use `add` for the types which are explicitly instantiated in `implemenation.cpp`. 
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

