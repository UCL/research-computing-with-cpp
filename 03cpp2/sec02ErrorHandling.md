---
title: Other Error Mechanisms
---

# Other Error Mechanisms

There are some other ways of handling errors that are worth being aware of outside of using exceptions. In this section we discuss _return codes_, which are particularly common in C-based external libraries and legacy code from earlier versions of C++, and `std::optional`, a special type which can represent the absence of a value. 

## Return Codes

Return codes are common in programming languages like C which do not have exceptions, and so will be frequently encountered by C++ programmers when using libraries built in C. (C is a very common languages library because C has become the lingua franca of programming languages, with most popular languages having a way to interface with C code.) 

The return codes approach can also be useful in its own right in C++, when we don't want to interrupt the program flow in the same way, or when errors are common and the exception overhead starts to become high. High frequencies of exceptions can cause particular problems for multi-threaded programming (you can read about [an example here](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2544r0.html) if you are interested).

In the return code approach, you define a function as returning an `int`, and the convention is to return `0` for a successful execution. Any other number encodes an error of some kind, and you can thus assign different meanings to different integers. Because we've used the return value to report success/failure, the meaningful output of the function is placed into a mutable argument (reference or pointer). This is one of the disadvantages of this approach, as it obsfuscates the logical structure of the function. 

Consider this example for getting the first element of a vector (which may be empty):
```cpp
#include <vector>
#include <iostream> 

using std::vector;

int head(vector<int> v, int &x)
{
    if(v.empty())
    {
        return 1;   // error code
    }
    else
    {
        x = v[0];
        return 0;   // success code
    }
}

int main()
{
    vector<int> v{5, 9, 4, 3};
    int x;
    if(!head(v, x))
    {
        std::cout << "It was empty." << std::endl;
    }
    else
    {
        std::cout << x << std::endl;
    }

    return 0;
}
```
- In this approach we have to declare our variable to hold the value first, and then pass it into the function. 
- We need to check the output of the function to see if it is successful. Unlike an exception, if we need to handle it further up the call stack then we have to check it separately **at every level until we reach the calling scope where we can handle it**. If functions are nested, then every intermediate function between where the error occurs and where it is handled needs to check for an error, and then return early with its own error code to be checked by the function that called it. This means that we often end up with a lot more error checking code using this approach. 
- Assigning appropriate error codes can be tricky, and you need to remember what they all mean. Using an accessible namespace to declare some `const` variables to give meaningful names to different error codes can be useful here, but beware that library code that you interface with will use their own conventions! 


## `std::optional`

Some functions might make most sense to conceptualise as returning either a _value_ or _nothing_. Examples might be getting the first element of a list (either the first element of a list or nothing if the list is empty), or looking up a value in a key-value map (return the value if the key is in the map or nothing if it isn't). These can be handled using exceptions or error codes of course, but a special _nothing_ value can be useful sometimes to explicitly represent this value. 
- Having a special _nothing_ value can help to avoid unitialised variables. This can happen for example when we want to assign a new variable from some function which may fail.
- A _nothing_ value can also be very useful for class design. For example, if you have a class representing some student data, we could set a grade for a course to be either a _value_ (the mark awarded) or _nothing_ (no mark is given yet). This avoids potentially misleading data using default values. 
    - It can also be very useful for avoiding null pointers when objects need to point to other objects, but we'll discuss pointers next week! 
- A _nothing_ value can be propagated through the rest of your calculations like any other value. If your code properly handles the _nothing_ values, this can sometimes simplify the control flow of your program by allowing all your data to go through the same pipeline. 

In C++ this approach can be handled using `std::optional` (in C++17 onwards); it is similar to the concept of the _Maybe monad_ in Haskell and a number of other similar structures in various languages. 

Like `vector`, `std::optional` uses angle brackets to declare the type of value that it can hold. `std::optional` can either hold a value, or the special value `std::nullopt`. 

The following code uses `std::optional` to get the first element (or "head") of a `vector`.
```cpp
#include <iostream>
#include <optional>
#include <vector>

using std::vector;
using std::optional;
using std::nullopt;

optional<int> head(const vector<int> &v)
{
    return v.size() > 0 ? optional<int>(v[0]) : nullopt;
}


// This function defines the << operator for streaming an std::option
// to an output stream such as std::cout.  
std::ostream& operator<<(std::ostream &os, optional<int> x)
{
    if(x)   // this is defined as: "if x has a value"
    {
        os << x.value();
    } 
    else    // otherwise it must be nullopt
    {
        os << "nothing";
    }
    return os;
} 

int main()
{
    vector<int> v1{5,9,4,3};
    optional<int> x = head(v);
    
    vector<int> v2;
    optional<int> y = head(v2);

    std::cout << x << ", " << y << std::endl;

    return 0;
}
```
The output for this program will be
```
5, nothing
```
