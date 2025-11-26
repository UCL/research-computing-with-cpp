---
title: C++ Syntax
---


# C++ Syntax

Let's dive a litte more into C++ syntax by looking at how to declare variables, call functions, and make use of control structures. First of all though, we should take a quick look at _types_ in C++, since these are going to come up a lot. 

## Types

We've already come across, `int`, `float`, `bool` and other keywords that define the *type* of variables and parameters. C++ is a *statically-typed* language, so the *types* of every single variable or function parameter must be known at *compile time*, and those types cannot change during runtime. 

> **Compile time** refers to the time at which the code is compiled. It's used in contrast to **runtime** (or **run time**) which refers to the time while the programming is actually running. For example, if a program requires a number to do some computation and you can write that number in the source code itself, that number is known at *compile time*. If, say, the user needs to input that number, the value is only known at *run time*.

This is in contrast to the *dynamically-typed* Python where we can define a variable `x` and assign it the integer value `2`, then reassign a string `"2"` to the same variable!

```python
x = 2
x = "2"
```

In C++ this kind of code will produce a compiler error.

We'll explore later what types are available in C++ (and how we can create our own) but a useful initial list of basic types is:

- `bool`: a boolean value, i.e. `true` or `false`
- `int`: an integer value, e.g. `-4, 0, 100`
- `float`: a 32-bit floating-point value `-0.2, 0.0, 1.222, 2e-3`
- `double`: a 64-bit floating-point value (same as `float` but can represent a greater range and precision of real numbers)
- `char`: a single character, e.g. `'a', 'l', ';'`

The _stardard library_ also defines many important types that we will use very frequently in C++; these are made available through `#include` statements and are prefixed by `std::` because they are part of the standard library _namespace_. Some common examples are given below including the necessary includes:

- `std::size_t`: stands for "size type", and is machine dependent but will likely be 64 bits (8 bytes) on the machines you work on. It is unsigned (i.e. only represents values $\ge 0$) and is large enough to handle the largest size that an object can be in C++ on a given architecture; as such it is commonly used for indexing arrays and other data structures. 
  - `#include<cstddef>` 
- `std::string`: text represented as a string of characters. Characters in a string can be iterated over similarly to lists. 
  - `#include<string>`
- `std::array<T, n>`: a kind of array with `n` elements of type `T`, e.g. `std::array<double, 3>` is an array of three doubles which could be used to represent a position in 3D space $(x, y, z)$. The size of the array must be known at compile time and cannot change.
  - `#include<array>`
- `std::vector<T>`: a kind of array of elements of type `T`, e.g. `std::vector<int> {1, 100, -1}` declares a vector of integers. The size of a vector can be determined at runtime, and it can also grow and shrink as desired. 
  - `#include<vector>`

## Declaring and Assigning Variables

A variable is declared by first declaring its _type_, then its _name_, and then (optionally) its _value_. A variable which is not assigned a value may be _uninitialised_ and contain an unknown value, so this should generally be avoided where possible. 

Some examples are given below; not the use of the `#include` and `std::` for the `string` type. 

```cpp
#include<string>

int main() 
{
  int x = 5;
  double y = 7.2;
  std::string greeting = "Hello!";
}
```

Some types in C++ can be _implicitly_ converted, but it is important to understand what is really happening here. Take for example, this code (we will omit the boilerplate for brevity):

```cpp
double y = 7.2;
int x = 5;

y = x;  // What happens here?
```

The assignment `y = x` takes the _value_ of `x` (an `int` representation of 5) and converts it to a `double` representation of the number 5, and then assigns that value to `y`. *The types of `x` and `y` have not changed.*  `x` is still an integer `5` and `y` is now a `double` 5.0. 

This kind of implicit conversion happens quite commonly, take for example:

```cpp
double y = 7;
```

Here the compiler will interpret the literal `7` as an `int` and then convert it to `double` to be assigned to the variable `y`. 

## Calling Functions

Like most common languages, functions are called like:
```cpp
int main() 
{
  int x = 5;
  string x_as_string = to_string(x);
  string three_as_string = to_string(3);
}
```
where the *return value* is, here, assigned to the variable `x_as_string`. We can call functions without keeping the return value by just not assigning the function call:
```cpp
int main() 
{
  to_string(3);
}
```

## Conditional logic using `if`/`else` statements

A true/false value is called a Boolean value, or `bool`. Conditional statements test the value of a Boolean value or expression and execute the following code block if it is `true`. (Remember that a code block is contained within curly braces `{}`, and can be as large as you like.)

```cpp=
// if statement with a Boolean variable
if(condition)
{
    std::cout << "Condition was true!" << std::endl;
}

// if statement with a Boolean expression
if(x < 10)
{
    std::cout << "x is too small." << std::endl;
}

// can also be a function which returns a bool!
if(f(x))
{
    std::cout << "f(x) was true!" << std::endl;
}
```
In the examples above, nothing will happen if the statement inside the brackets is not true. 

If you want something to happen when the statement is false, you can also use `else` and/or `else if` statements.

```cpp=
if(x < 10)
{
    std::cout << "x is small" << std::endl;
}
else if(x > 50)
{
    std::cout << "x is large" << std::endl;
}
else
{
    std::cout << "x is neither large nor small." << std::endl;
}
```

## Loops (`for` and `while`)

```cpp=
for(unsigned int i = 0; i < 100; ++i)
{
    // loop code goes here
}
```
- The brackets after the `for` set up three things:
    - first we declare a variable, if any, that we want to use for the loop.
    - next we have the loop condition; the loop continues while this is still true. 
    - finally we have a statement which should execute at the end of each loop iteration. 
    - In this case, we execute the loop 100 times, with `i` taking the values `0` to `99`. 
    - The variable `i` is available inside the loop. 
- `unsigned int` is a type for _unsigned integers_, which are integers that cannot be negative. It's a good idea to use these for counting and other values which shouldn't be less than 0. You can also use `std::size_t`. 
- `++i` increments the value of `i` by 1. 

If we have a `vector` or similar container, we can loop over its elements without writing our own loop conditions:
```cpp=
#include <vector>

int main()
{

    std::vector<int> v(10); // declare a vector of ints with 10 elements.

    for(int &x : v)
    {
        std::cout << x << std::endl;
    }
    
}
```
- The `for` loop iterates over the elements of `v`. 
- At each iteration, the variable `x` is given the value of the current element.
- This is a good way to iterate over containers when we don't need to refer to indices explicitly, as it avoids possible programmer errors! 

`while` loops have simpler syntax than `for` loops; they depend only on a condition, and the code block executes over and over until the condition is met. This is useful for situations where the number of iterations is not clear from the outset, for example running an iterative method until some convergence criterion is met. 

```cpp=
while( (x_new - x_old) > 0.1)  // convergence criterion
{
    x_old = x_new;
    x_new = f(x_old);    //iteratively call function f on x until it converges
}
```