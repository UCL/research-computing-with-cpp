---
title: C++ Standard Library
---

## C++ Standard Library

The C++ standard library is a collection of data-structures and methods which must be provided with any standard-compliant implementation of C++. As a result, using the standard library is portable across different systems and compilers, and does not require downloading and linking additional libraries (which will be a topic we cover in a later week). It does, however, require the use of header files, as we'll see in a moment. 

The C++ language standard is always evolving, with the most recent version of the standard being C++20 (released 2020), and the next planned to come some time this year (C++23). Sometimes there may be requirements to work with a specific C++ standard (more often, a minimum standard), and so it can be important to check whether a feature that you want to use is available to you in the version of C++ that you will be using. 

In this course the language features that we will make use of will be compatible with C++14 onwards, though we will use C++17 as our standard. This week we will go over some of the most commonly used components of the standard library, but this will only scratch the surface and becoming familiar with the breadth of the library takes time and practice. 

## Input / Output

Input and output (often abbreviated to IO or I/O) to the terminal or to a file is important for interacting with users and retrieving the results of programs. It can also be useful for some debugging purposes, although you should use a debugger where you can (more on that in week 6). The standard library has some useful headers for this; we'll make heavy use of `<iostream>` to write to, or receive input from, the terminal. 

### Output 

```cpp
#include <iostream>

int main()
{
   std::cout << "Hello World!" << std::endl;

   return 0;
}
```
This short program prints `Hello World!` to the terminal.
- We use `std::` to indicate that a function is part of the `std` (standard) namespace, which prevents named functions, classes, and variables in the library from clashing with our own. 
- `std::cout` is a stream: this is an output buffer than can take a variety of types including strings, integers, and floating point numbers, and turns them into output. 
- `std::endl` is a special marker for the line end in the output stream. 

### Input
```cpp
#include <iostream>
#include <string>

int main()
{
   std::string name;   

   std::cout << "What is your name?" << std::endl;

   std::cin >> name;

   std::cout << "Hello " << name << "!" << std::endl;

   return 0;
}
```
- `cin` is an input buffer.
- With `cin` the arrows go the other way, because the information passes from the input buffer into the `name` variable. 
- There are other kinds of input buffers, like reading from files. 

## Containers

Containers are an important part of the C++ standard library; these allow us to keep collections of objects such as lists (`vector`, `array`), sets (`set`), or maps (`map`, `unordered_map`) of key-value pairs, among others. These are some of the most common classes that you will use in C++ programming, so it is a good idea to familiarise yourself with them. We'll discuss `vector` as an example here, but see the section "Using C++ Documentation" for more information on how to learn about the other kinds of containers. 

### Vector

The vector class is defined in the `<vector>` header. It is used when you want to keep a list of elements. 

```cpp
#include <vector>

int main()
{
    std::vector<int> fibbonacciList = {1, 1, 2, 3, 5, 8, 13};

    return 0;
}
```
- We must include `<vector>` to use the vector class.
- When we declare a vector we also must place the type of object that the vector holds in angle brackets. So a vector of integers is declared `std::vector<int>`, and a vector of doubles is declared `std::vector<double>`. 
- We can declare an empty list using `std::vector<int> myIntList`, or we can use the curly-brace notation above to give some initial values. 

Vectors are dynamically sized, which means we change, add, or remove elements. 
```cpp
std::vector<int> v = {1,2,3};
v.push_back(4);  // Add an element to the end of our vector 
v.pop_back();    // Remove last element from vector

v.insert(v.end() - 1, 99);  // Insert an element before the last element of the vector
v.erase(v.begin() + 1);     // Remove the second element of a vector
```
- `v.end()` and `v.begin()` return iterators, special types which can be used to iterate over containers. The methods `begin()` and `end()` return iterators to the start and end of a container respectively, and iterations can be incremented using regular arithmetic. These can also be used for looping e.g. `for(vector<int>::iterator it = myNums.begin(); it != myNums.end(); it++)`. You can use `auto` to infer the iterator type in a loop declaration to make it more succinct.
- THe memory for a `vector` is assigned in a contiguous block. This helps with performance because the vector class can find the memory location of a given element using pointer arithmetic.
- A `vector` doesn't have a fixed size, but it does have a certain amount of memory allocated to it. If you add enough elements to outgrow this size, it will have to allocate new memory of a larger size, and copy the elements over to this new memory to keep it contiguous. The cost of this operation scales linearly with the size of your vector; you cannot necessarily predict when this will happen as the amount of memory allocated to a vector by default will be compiler dependent. 
- If you know how much space you will need for a vector, or can place an appropriate limit on it, then you can use the `.reserve(int)` method to reserve space for for a given number of elements, e.g. `v.reserve(20)` will allocate enough memory to `v` to store 20 elements. 

We can also access elements of a vector in a couple of different ways.
```cpp
#include <vector>

int main()
{
    std::vector<int> fibbonacciList = {1, 1, 2, 3, 5, 8, 13};

    int element0 = fibbonacciList.at(0);
    int element3 = fibbonacciList[3];

    return 0;
}
```
- Indexing of vectors (and other array like containers) always starts at 0.
- We can use the `.at(i)` method on a vector to get the value at that position in the vector. If `i` is outside of the range of the vector then this method will throw an error, which can be handled by your program. (See next week when we discuss exceptions.)
- We can also access a vector using square brackets: `myVector[i]`. This is substantially faster than using `myVector.at(i)`, but does no bounds checking. If `i` is outside of the size of vector then the behaviour is *undefined*: this means the standard does not define what the program should do in this case, and the result will be compiler dependent. *Undefined behaviour should always be avoided. Only use square brackets if array access performance is important and you can be sure that you will not go outside of the bounds of the vector.*
- Accessing and modifying vector elements using `[]` has high performance, and can even out-perform traditional C-arrays if compiled with optimisations turned on, which makes `vector` an excellent general purpose class for lists. 

We can use *range based loops* with vectors, since the vector object keeps track of the number of elements within it. We could do this e.g. to print out all the elements of a vector.
```cpp
#include <vector>

int main()
{
    std::vector<int> fibbonacciList = {1, 1, 2, 3, 5, 8, 13};

    for(auto num : fibbonacciList)
    {
        std::cout << num << std::endl;
    }

    return 0;
}
```
- Declaring a range based loop we use the syntax: `for(type element : list){ loop code }`.
- Here we have used the `auto` keyword to ask C++ to infer the type for us; this can be useful when we want to avoid writing out lots of types explicitly!
- This will iterate through all the elements of the list, in order. For each loop iteration, `num` will be initialised with the data for that element in the list. 
- `num` is initialised the same way as function arguments, so by default the elements of the list are passed by value i.e. a copy of the element is made and stored that in the variable `num`. This means any changes to `num` in the loop code are not reflected in the vector itself. If you want to avoid copy overheads or make changes to your vector, you should make this variable a reference using the `&` operator just like when passing by reference to a function (see Passing by Value & Passing by Reference). In this case we would write `for(auto &num : fibbonacciList)` and our loop code would receive a reference to each element of the list rather than a copy.  
- Range based loops can be used with other containers which are iterable, like `array` and `map`. 
- You can also iterate through these containers using traditional `for` loops and iterator types or integers. 

Traditional `for` loops can be useful for iterating through vectors when you need to keep track of the index of an element, for example when assigning a vector a value that depends on its index or when working with multiple vectors at once. 
```cpp
#include <vector>

int main()
{
    std::vector<int> a = {1, 2, 3, 4};
    std::vector<int> b = {5, 6, 7, 8};
    std::vector<int> c(4); // create a vector with size 4

    for(uint i = 0; i < 4; i++)
    {
        c.at(i) = a.at(i) + b.at(i);
    }

    return 0;
}
```

## Algorithm

The `<algorithm>` library is an important part of modern C++ code. It contains a variety of commonly used algorithms, many of which operate on containers. These include sorting, searching, counting, maxima, minima, and merge. If there is some functionality that you need then it is usually worth checking if there is a standard library implementation for it already, before trying to implement it yourself or seeking third party libraries. This will save you time on implementation and testing, and keep your code portable. 

The functionality in `algorithm` can be made much more flexible and interesting by combining with other functions, as many of the functions in `algorithm` take functions as arguments. One way to do this is to write a function and pass a reference to it. Imagine that we want to count the number of even numbers in a vector of integers. Using `count_if` from the `<algorithm>` library we could write:
```cpp
    bool isEven(int x)
    {
        return x % 2 == 0;
    }

    int main()
    {
        vector<int> myNums = {1, 6, 5, 8, 3, 5, 4, 2, 8, 9, 9, 7, 6};
        int numEvens = std::count_if(myNums.begin(), myNums.end(), &isEven)
        std::cout << "num evens = " << numEvens << std::endl; 

        return 0;
    }
```
- Note that here we give `count_if` iterators for the beginning and end of the vector.
- The third argument given to `count_if` is a reference to a function which takes an integer `x` and returns `true` if `x` is divisible by two. 
- `count_if` applies the function to each element of the vector and adds to the count only if it evaluates to true. 


## Anonymous Functions

The code above using `count_if` works fine, but if we have many such algorithm calls, which employ different tests, we can end up with code that has a lot of function definitions that are only used once. We can express this more succinctly using *anonymous functions*, also known as *lambda expressions*. These are function objects which can be defined without the usual C++ function declaration, and passed around like ordinary objects (including being copied). Compare the following code with the sample above:
```cpp
    int main()
    {
        vector<int> myNums = {1, 6, 5, 8, 3, 5, 4, 2, 8, 9, 9, 7, 6};
        int numEvens = std::count_if(myNums.begin(), myNums.end(), [](int x){return x%2 == 0;})
        std::cout << "num evens = " << numEvens << std::endl; 

        return 0;
    }
```
This code achieved the same thing, but there is no named function passed to `count_if`. Instead, we have an anonymous function - an object that we have created for the purposes of providing an argument to `count_if`. We can also use lambda expressions to create named objects, like so:
```cpp
auto isEven = [](int x){return x%2 == 0;};
```
The object `isEven` can now be passed to functions, copied, overwritten, and so on. 

Lambda expression syntax can look a little confusing at first, but becomes simpler if we understand what the three different kinds of brackets are for:
- [] Square brackets are for variables to be captured from the environment. These are variables which you want to be available in your function, but are not explicit function arguments. For example, you might want to apply a lambda function to every element of a vector (in which case the vector elements are the function arguments) but also take into account another variable in the environment that will be the same for each function application. 
- () Round brackets are for function arguments. These are passed into the anonymous function just like arguments are passed to any other function. 
- {} Curly braces are used to contain the function execution code. This may refer to the variables passed as arguments or captured from the environment. It can involve multiple lines / statements, separated by semicolons (`;`) just like normal code, although lambda expressions are usually used for short code fragments. If there is no return statement then the return type is `void`, as usual. 

We can see from our previous example the use of the `()` and `{}` brackets to define our function argument `(int x)` and our function body `{return x%2 == 0;}`. So far our variable capture `[]` has been empty, so let's modify our function to make use of this feature. Say we want to count the number of elements in my list, divisible by some number `n`, which we won't know ahead of time. We can write 
```cpp
    int n = getSomeNumber();  // Don't know the number n at compile time
    int numDivisibleN = std::count_if(myNums.begin(), myNums.end(), [n](int x){return x%n == 0;});
    std::cout << "Number divisible by n = " << numDivisibleN << std::endl;
```
- We use variable capture `[n]` to get the divisor `n` from the the environment and bind it to our function. Now when we call the function on each element (the argument `x` in round brackets) we divide each element by `n` and get the remainder!
- This variable capture behaviour is harder to replicate using function references, so lambda expressions are extremely useful for this kind of code.  

## Using C++ documentation 

You will often find when programming, especially in a language with such an expansive standard library, that there are things that you need to look up. There are a large number of classes and functions available to C++ programmers, many of which may be new to you or require refreshing at various points. 

Two common sites for C++ refernce are:
- https://cplusplus.com/
- https://en.cppreference.com/ 

Both are extremely useful and give thorough information about how to use classes and methods according to the specification in the C++ standard. Take for example the pages on the `vector` class: https://cplusplus.com/reference/vector/vector/ or https://en.cppreference.com/w/cpp/container/vector. Each tells us:
- which header the class is found in 
- a summary of the class and how it is stored in memory
- information on performance of basic operations 
- a list of member functions and other information about the class (with links to detailed information and code samples)
- a list of non member functions which operate on the class 

You can also use these reference sites too look up libraries that make up part of the C++ standard. For example, https://cplusplus.com/reference/algorithm/ contains information about what is contained in the `<algorithm>` library that we briefly discussed above, with links to more detailed information about the functions available and the container types on which they operate (https://cplusplus.com/reference/stl/). 

You should familiarise yourself with these resources to make the best use of C++ in the future. When writing research code, try to get into the habit of checking whether the C++ standard, or common libraries, already implement the functionality that you need satisfactorily before trying to implement your own. Online searches and forums such as StackOverflow can be a useful resource for this as well, but remember to check the official specification if you do find a class or function that you think will be useful to you, to be sure that you have accurate information about it! 