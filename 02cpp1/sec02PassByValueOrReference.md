---
title: Pass by Value and Pass by Reference
---

Estimated Reading Time: 15 minutes

# Pass by Value and Pass by Reference 

Variables can be passed to functions in two ways in C++: "pass by value" and "pass by reference". This choice significantly alters both the behaviour and the performance of a function. 

## Pass by Value

Pass by Value means that we copy the value of the variable we want to pass into the function, and the function works on this copy and leaves the original alone. Any changes that the function makes to the variable will not affect the value of that variable once you leave the function's scope. To pass by value you simply write the type and name of the variable in the function parameters in the usual way. 
```cpp
int add(int a, int b)
{
    return (a + b);
}
```
This function can be safer, but is not time or memory efficient if variables are complex or large in size. You should only use pass by value for large pieces of data if you need an explicit copy made to work on locally in the function body. 

## Pass by Reference 

Pass by Reference means that we tell the function where the original variable has been stored in memory, and we allow the function to work directly with that original variable. This has two major consequences:
   - We only pass a memory address -- 4 bytes, the same size as an integer -- to the function, so there is no additional memory allocated to copy the object.
   - The original variable can be changed by the function, and so any changes that happen within the function are retained after we leave the function's scope. 

We indicate that we want a reference to a variable using the `&` operator. The function below will take a reference to an integer and increment that integer by one. Because we have changed the value stored at that memory location, once we leave this function the variable that we passed to it will retain this increased value. 
```cpp
void increment(int &x)
{
    x = x + 1;
}
```
Passing by reference can save significant time and memory by avoiding making needless copies of variables, but at the cost of making variables potentially vulnerable to being changed by a function. This makes it harder for someone using the function to reason about the program, and what the value of the variables they pass in will be once the function has finished.  

## Using `const` in Pass By Reference

We can retain the performance advantages of pass by reference and still protect our variables from changes by passing a const reference. 
```cpp
void constRefExample(int const &x)
{
    return 5*x; 
}
```
The declaration `int const &x` means that `x` is a reference (`&`) to a constant (`const`) integer (`int`). This means that the integer value cannot be changed, and so any attempt to change the value of `x` in the function will lead to a compiler error. 

## Using references for output variables

When we use a `return` statement in a function, we are also passing by value, and a copy of the variable that we are returning is made. Just like with inputs to a function, this can be a performance issue if the data that we want to output is large. There are however a few things to note about the efficiency of `return` statements:
- Objects are copied using their _copy constructor_, a special function in their class definition which defines how to create a new object and copy the current object's data. (This may be automatically created by the compiler.)
- Some objects also have a _move constructor_ defined, in which the data is not explicitly copied, but a new object takes control of the data. We'll return to this idea when we talk about pointers in the next section! (The move constructor may also be automatically created by the compiler.) 
- For such classes an object can be returned without making a copy, since the compiler knows that the original object will immediately go out of scope (and then be destroyed) and can therefore have its data moved without any conflict over which object owns the data. (This is why this optimisation can be used when returning a value but _not_ when passing an object to a function by value: when passing an object to a function the original object will continue to exist.)
- **The compiler will use a move constructor whenever available, and a copy constructor when not.** Therefore, you may find that returning values is more performant than you expect from the size of the data-structure. 

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

Instead of declaring a variable and setting it equal to the return value of a function, we can instead declare the variable and then pass it into the function by reference. 
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
- Pass larger arguments (> 4 bytes) by `const` reference if you can.
- Pass by (non const) reference if you need to work on a variable in place i.e. the function should change the value of the argument itself. 
- Avoid `return` with large data-structures for the same reason. These should be passed in and out by reference as function arguments. 
- **N.B.** If passing by reference, you can only pass literals (values like numbers and strings which are not assigned to a variable) if using a `const` reference. Consider two function signatures `refAdd(int &a, int &b)` and `constRefAdd(const int &a, const int &b)`: we can call `constRefAdd(5, 12)` just fine, but if we call `refAdd(5, 12)` we will get an error. 
- **N.B.** Never use a `return` statement to return a reference (or a pointer) to a local variable e.g. `return &x;` as the local variable will be destroyed when we leave the function scope. This will lead to a segmentation fault (memory error). 
