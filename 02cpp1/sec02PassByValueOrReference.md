---
title: Pass by Value and Pass by Reference
---

Estimated Reading Time: 15 minutes

# Pass by Value and Pass by Reference 
 
Variables are handles for data, which needs to be stored in memory somewhere so that it can be read and modified. 

In C++ we can find out the memory address of a given variable using the `&` operator, as in the following code snippet:

```cpp
#include <iostream>

int main()
{
    int x = 15;
    std::cout << "x is stored at address " << &x << " and has value " << x << std::endl;
    x += 1;  // Increment the value of x by 1. 
    std::cout << "x is stored at address " << &x << " and has value " << x << std::endl;
    return 0;
}
```
- Declaring `int x = 15` tells the compiler that we will need to store some data which is to be interpreted as an integer with an initial value of 15. The compiler then knows:
    - How much memory to allocate for the variable (for an `int` this is normally 4 bytes).
    - How the variable can be used in the remainder of the program (see the notes on types for more information).
    - To set the data at that memory location to be equal to `15` at this point in the program. 
    - From now on, in this scope, `x` will refer to the value stored at this memory location, and changes to `x` will lead to changes to the data stored there. 
- `&x` gives us the _address_ in memory where the value for `x` is stored. An address is a unique (usually 8 byte) numeric value for each memory location where we can store data. Changes to the value of `x` do not change its address: **the address of a variable cannot be changed**.
- The compiler will select an unoccupied memory address to store the variable in for you. 

The output of this program is 
```
x is stored at address 0x7ffe13999ed4 and has value 15
x is stored at address 0x7ffe13999ed4 and has value 16
```
- The value of the address is given in [hexadecimal](https://en.wikipedia.org/wiki/Hexadecimal), which is prefixed by `0x` when printed.
- The address will be system dependent but note that the address does not change, but the value does. 

Given this, we might wonder what it means to "pass a variable to a function". Does the function need the _value_ that the variable represents, or the _location_ where the variable is stored?

There are broadly two approaches to giving functions access to variables from the calling scope:
1. The function has a _copy_ of the variable's _value_ in a new variable, independent from the original. This means read/write operations to this variable in the function operate on a separate memory location which does not affect the original. Once we return from the function scope, the original variable is unchanged because the function did not have access to the memory location where it is stored. 
2. The function has a _reference_ to the variable, which is to say it is given the memory location where the variable is stored. Any read/write operations are made to this same location, and therefore when we leave the function scope any changes made to the variable inside the function will still remain. 

Programming languages vary in their approaches to passing variables, but C++ gives us the choice when we define a function to pass each variable in either way: the first is called **pass by value** and the second is **pass by reference**. This choice significantly alters both the behaviour and the performance of a function in ways that we shall detail in the following sections.

## Pass by Value

Pass by Value means that we copy the value of the variable we want to pass into the function, and the function works on this copy and leaves the original alone. Any changes that the function makes to the variable will not affect the value of that variable once you leave the function's scope. In C++, passing by value is the default, so to pass by value you simply write the type and name of the variable in the function parameters in the usual way.

```cpp
int add(int a, int b)
{
    return (a + b);
}
```
This function can be safer, but is not time or memory efficient if variables are complex or large in size as the values need to be copied to new memory locations. You should only use pass by value for large pieces of data if you need an explicit copy made to work on and change locally in the function body but you cannot allow the function to change the original. 

We can see this call by value in action explicitly in the following code example by checking the address and values of variables:
```cpp
#include <iostream>

using namespace std;

int add(int a, int b)
{
    cout << "In add function, before adding." << endl;
    cout << "a is stored at address " << &a << " with value " << a << endl;
    cout << "b is stored at address " << &b << " with value " << b << endl;

    a = (a + b);

    cout << "In add function, after adding." << endl;
    cout << "a is stored at address " << &a << " with value " << a << endl;
    cout << "b is stored at address " << &b << " with value " << b << endl;

    return a;
}

int main()
{
    int x = 12;
    int y = 9;

    cout << "Before add function." << endl;
    cout << "x is stored at address " << &x << " with value " << x << endl;
    cout << "y is stored at address " << &y << " with value " << y << endl;

    int z = add(x, y);

    cout << "After add function." << endl;
    cout << "x is stored at address " << &x << " with value " << x << endl;
    cout << "y is stored at address " << &y << " with value " << y << endl;
    cout << "z is stored at address " << &z << " with value " << z << endl;

    return 0;
}
```
Outputs:
```
Before add function.
x is stored at address 0x7ffee8b8b32c with value 12
y is stored at address 0x7ffee8b8b330 with value 9
In add function, before adding.
a is stored at address 0x7ffee8b8b30c with value 12
b is stored at address 0x7ffee8b8b308 with value 9
In add function, after adding.
a is stored at address 0x7ffee8b8b30c with value 21
b is stored at address 0x7ffee8b8b308 with value 9
After add function.
x is stored at address 0x7ffee8b8b32c with value 12
y is stored at address 0x7ffee8b8b330 with value 9
z is stored at address 0x7ffee8b8b334 with value 21
```
- Note that the variables `x` and `y` are passed to the `add` function to serve as arguments `a` and `b` respectively.
- In the function we can see that `a` and `b` are variables which are stored a separate memory locations from `x` and `y`. When we modify `a = a + b` the value stored for `a` changes, but `x` and `y` do not because their memory locations were untouched. 

## Pass by Reference 

Pass by Reference means that we tell the function where the original variable has been stored in memory, and we allow the function to work directly with that original variable. This has two major consequences:

- We only pass a memory address -- usually 8 bytes in a 64-bit system -- to the function, so there is no additional memory allocated to copy the object.
- The original variable can be changed by the function, and so any changes that happen within the function are retained after we leave the function's scope. 

We indicate that we want a reference to a variable using the `&` symbol after the type of the argument in the function signature. The function below will take a reference to an integer and increment that integer by one. Because we have changed the value stored at that memory location, once we leave this function the variable that we passed to it will retain this increased value. 

```cpp
void increment(int &x)
{
    x = x + 1;
}
```
- Even though we have passed `x` by reference, `x` is just an ordinary `int` variable; this is only telling the compiler to give the function access to the original memory address instead of copying it to a new one. So inside the function we can just use `x` as normal.  
    - A reference just means an alias (i.e. a new name) for an existing variable, and therefore has the same type as the existing variable.
- It can be a bit confusing that the notation for a reference is the same as the address operator (`&`). These two uses of `&` come up in different contexts though:
    - When it is used in a type context, i.e. follows a typename, it means a _reference to a variable of that type_.
    - When it is used in an expression which evaluates to a value, e.g. `cout << &var << endl`, then it is the address operator and it evaluates to the memory address of the variable to which it is affixed. 


**Passing by reference can save significant time and memory by avoiding making needless copies of variables**, but at the cost of making variables potentially vulnerable to being changed by a function. This can make it harder for someone using the function to reason about the program, and what the value of the variables they pass in will be once the function has finished.  

We can once again illustrate this explicitly with addresses by re-using our example above, but changing the `add` function to take `a` and `b` by reference (`int &a` and `int &b`):
```cpp
#include <iostream>

using namespace std;

int add(int &a, int &b)
{
    cout << "In add function, before adding." << endl;
    cout << "a is stored at address " << &a << " with value " << a << endl;
    cout << "b is stored at address " << &b << " with value " << b << endl;

    a = (a + b);

    cout << "In add function, after adding." << endl;
    cout << "a is stored at address " << &a << " with value " << a << endl;
    cout << "b is stored at address " << &b << " with value " << b << endl;

    return a;
}

int main()
{
    int x = 12;
    int y = 9;

    cout << "Before add function." << endl;
    cout << "x is stored at address " << &x << " with value " << x << endl;
    cout << "y is stored at address " << &y << " with value " << y << endl;

    int z = add(x, y);

    cout << "After add function." << endl;
    cout << "x is stored at address " << &x << " with value " << x << endl;
    cout << "y is stored at address " << &y << " with value " << y << endl;
    cout << "z is stored at address " << &z << " with value " << z << endl;

    return 0;
}
```
Yielding:
```
Before add function.
x is stored at address 0x7ffdd308656c with value 12
y is stored at address 0x7ffdd3086570 with value 9
In add function, before adding.
a is stored at address 0x7ffdd308656c with value 12
b is stored at address 0x7ffdd3086570 with value 9
In add function, after adding.
a is stored at address 0x7ffdd308656c with value 21
b is stored at address 0x7ffdd3086570 with value 9
After add function.
x is stored at address 0x7ffdd308656c with value 21
y is stored at address 0x7ffdd3086570 with value 9
z is stored at address 0x7ffdd3086574 with value 21
```
- Note that now `a` has the same address as `x`, and `b` has the same address as `y`. 
- When `a` is updated, the value at its memory location is changed, so after the `add` function call we can see that the value of `x` has changed as well. 

## Using `const` in Pass By Reference

We can retain the performance advantages of pass by reference and **still protect our variables from changes** by passing a `const` reference. 

```cpp
void constRefExample(int const &x)
{
    return 5*x; 
}
```

The declaration `int const &x` means that `x` is a reference (`&`) to a constant (`const`) integer (`int`). This means that the integer value cannot be changed within this function, and so any attempt to change the value of `x` in the function will lead to a compiler error. 

Try writing a function where you pass an argument by const reference and try to modify it inside the function. Take note of what the compiler error looks like!

## Return Values

When we use a `return` statement in a function, we are also passing by value, although a copy of the variable is not necessarily always made. As with inputs to a function, this can be a performance issue if large output data ends up being copied. There are however a few things to note about the efficiency of `return` statements:

- Objects are copied using their _copy constructor_, a special function in their class definition which defines how to create a new object and copy the current object's data. (In many cases this can be automatically created by the compiler.)
- Some objects also have a _move constructor_ defined, in which the data is not explicitly copied, but a new object takes control of the data. We'll return to this idea when we talk about pointers later in the course. (The move constructor may also be automatically created by the compiler.) 
- Normally when an variable goes out of scope its memory is freed and can be reallocated to new variables. If we have a _local variable_ in the function scope that we want to return, we can't just give the address of the data (return by reference) because when the function returns the variable will go out of scope and that memory is freed. 
    - Although return types can be references, e.g. `int& someFunction()`, you have to be absolutely certain that the memory you are referencing will remain in scope. This could be e.g. a global variable, or a member of a class for an object which continues to exist. It should _never_ be a variable created locally in that function scope. Don't use reference return types unless you are really confident that you know what you are doing! 
- For classes with a move constructor a local object can be returned without making a copy, since the compiler knows that the object is about to be destroyed as soon as the function returns, and can therefore have its data transferred instead. (This is why this optimisation can be used when returning a value but _not_ when passing an object to a function by value: when passing an object to a function the original object will continue to exist.)
- **The compiler will use a move constructor when available if the object is deemed large enough for the move to be more efficient than a copy, and a copy constructor when not.** Therefore, you may find that returning values is more performant than you expect from the size of the data-structure. 

We can see this move or copy behaviour for return values in the following code example:
```cpp
#include <iostream>

using namespace std;

class Obj
{
    public:
    int a, b, c, d, e, f, g, h;

    Obj(int a, int b, int c, int d, int e, int f, int g, int h) : a(a), b(b), c(c), d(d), e(e), f(f), g(g), h(h) {}
};

Obj makeObj()
{
    Obj myObj(1, 2, 3, 4, 5, 6, 7, 8);
    cout << "In makeObj, myObj is at address " << &myObj << endl;
    cout << "myObj.a is at " << &myObj.a << endl;
    cout << "myObj.b is at " << &myObj.b << endl;
    return myObj;
}

int makeInt()
{
    int x = 5;
    cout << "In makeInt, x is at address " << &x << endl;
    return x;
}

int main()
{
    Obj newObj = makeObj();
    cout << "Outside the function, newObj is at address " << &newObj << endl;
    cout << "newObj.a is at " << &newObj.a << endl;
    cout << "newObj.b is at " << &newObj.b << endl;

    int y = makeInt();
    cout << "Outside the function, y is at address " << &y << endl;

    return 0;
}
```
Which yields the output:
```
In makeObj, myObj is at address 0x7ffdb4b6e0c0
myObj.a is at 0x7ffdb4b6e0c0
myObj.b is at 0x7ffdb4b6e0c4
Outside the function, newObj is at address 0x7ffdb4b6e0c0
newObj.a is at 0x7ffdb4b6e0c0
newObj.b is at 0x7ffdb4b6e0c4
In makeInt, x is at address 0x7ffdb4b6e094
Outside the function, y is at address 0x7ffdb4b6e0bc
```
- `Obj` is a large data type which contains 8 `int` values. (You'll see how to define these custom data types, called classes, in section 2.)
- In `Obj` there is no explicit copy or move constructor, these are implicitly filled in by the compiler for simple types like this. 
- Because `Obj` is large, the `makeObj` function returns the object using **move**. We can see that `myObj` and `newObj` have the same address. The control of this memory is moved from `myObj` to `newObj`, so when `myObj` goes out of scope and is destroyed the memory remains active and under the control of the new variable. 
- `makeInt` is identical in structure, but only returns a single `int`. There's no move defined for `int` because it is already so small; we can see that `x` and `y` have different addresses. Some small objects will also be copied instead of moved. 

**Note that not all types can be moved, and not all types can be copied.** In these cases, we can use references arguments as outputs. 

## Mutable References as Outputs

If a return statement has significant overheads, it may be avoided using references. Let's assume we have a large data class `ImmovableData` with no move constructor. 

```cpp
ImmoveableData GenerateData(const int &a)
{
    ImmovableData D;
    for(int i = 0; i < a; i++)
    {
        // Do some data generation
        ...
    }
    return D;
}

int main()
{
    ImmovableData data = GenerateData(100000);

    return 0;
}

```

- This code will create a large data-structure the function call, and then copy that structure when the function returns and place the result in the variable `data`. The original data-structure is then deleted.

Instead of declaring a variable and setting it equal to the return value of a function, we can instead declare the variable in the calling scope, and then pass it into the function by reference. 

```cpp
void GenerateInPlace(const int &a, ImmovableData &v)
{
    for(int i = 0; i < b; i++)
    {
        //Do some data generation
        ...
    }
}

int main()
{
    ImmovableData data;
    makeListInPlace(100000, data);

    return 0;
}
```

- In this case, only one data-structure is made. Its data is updated in the function, but it never has to be copied. Once we exit the function, the changes to `data` have persisted and we can use the values that we have assigned to it. 

## Which should I use?

- Passing small types like `int` or `float` by value is fine, as they are the same size as a reference.
- Passing by value is also fine if you need a copy of the argument to work on in the function body without affecting its value outside the function. 
- Pass larger arguments (> 8 bytes) by `const` reference if you can.
- Pass by (non const) reference if you need to work on a variable in place i.e. the function should change the value of the argument itself. 
- Avoid `return` with large _immovable_ data-structures for the same reason. These should be passed in and out by reference as function arguments. 

**Further things worth noting**:
- If passing by reference, you can only pass literals (values like numbers and strings which are not assigned to a variable) if using a `const` reference. Consider two function signatures `refAdd(int &a, int &b)` and `constRefAdd(const int &a, const int &b)`: we can call `constRefAdd(5, 12)` just fine, but if we call `refAdd(5, 12)` we will get an error. 
- Never use a `return` statement to return a reference (or a pointer) to a local variable e.g. `return &x;` as the local variable will be destroyed when we leave the function scope. This will lead to a segmentation fault (memory error). 
- You can return by reference a variable which is not local to that function's scope, for example a member function in a class may return a member variable of that class by reference, since when the function ends the object and its data will still exist. However, you must be sure that you will not keep the reference to the data for longer than the object's lifetime; if the object passes out of scope and you continue to try to use the reference then you will have a memory error.  
