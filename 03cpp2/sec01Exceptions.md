---
title: Exceptions
---

Estimated Reading Time: 40 minutes

# Exceptions 

Exceptions are used for error handling in C++ (and many other languages). The standard library provides a variety of exceptions that we can use straight away, although we can also write our own exceptions if we want additional functionality. 

The `<stdexcept>` header contains the following exception types:

- `logic_error`
- `invalid_argument`
- `domain_error`
- `length_error`
- `out_of_range`
- `runtime_error`
- `range_error` 
- `overflow_error`
- `underflow_error`

**All of these exception classes inherit from the class `std::exception`.** Inheritance will play an important role in how we define, identify, and handle exceptions.  

You may notice these exceptions being thrown by a number of different methods that you use in C++! For example, if you try to access a `vector` using `.at(i)` outside of the range of the vector, an `out_of_range` exception will be thrown, which will halt program execution and be reported in the terminal output if you don't handle it properly. 

When you look up a function in the C++ documentation, you can see what exceptions it can throw, and therefore what kinds of errors you may need to consider checking for. Some functions have "no throw" guarantees, which means that they cannot throw exceptions. Just because a function does not throw exceptions does not mean it is impossible for an error to occur: be sure to check if the function is using some other method of reporting and handling errors. For example, some code which is written in, or compatible with, C or FORTRAN will use the return value of the function or a mutable reference parameter to report whether the function execution was successful or not.

## The basic idea of Exceptions

An exception is a type of object which can be "thrown" when something happens that is not supposed to happen, but which has been anticipated by the programmer as a possibility. The exception is then "caught" in another part of the code which then handles the error. Uncaught exceptions will terminate the program execution (which may be the desired effect if there is no way to continue after an error). 

For example, if you write a function which requires the use of the first three elements of a vector, your function should check that the vector passed to it has at least three elements. One way of handling the case where it does not have enough elements is to throw an exception: this halts the execution of the function (thereby preventing any attempts to, for example, access elements that aren't there), and returns control to the calling code. If the code which called the function has been set up to detect exceptions, then it can "catch" the exception and handle it appropriately. If the exception is not caught by the calling function, then that function halts as well and the exception propagates up the stack to that function's caller, and so on until it is either caught or we reach the end of the call stack and the program terminates. 

**N.B.** Exceptions should not be used a method of controlling program flow and should only be used to cover unusual cases that shouldn't normally occur (hence "_exception_"). **Do not use throw exceptions as an alternative way of returning a value from a function or as a way to exit a loop.** These should always be handled using `return` statements (for function return) and `break` statements (for loop exit). Exception handling statements should be used to handle cases where normal execution of the program simply cannot continue, and in a typical run of the code the exception should not be thrown. Some good examples of where to use exceptions for error handling are: running out of space for a process that needs to allocate memory, file system input/output errors, runtime errors from users giving unexpected bad input. Using exceptions to control program flow other than error handling can be a problem for two reasons:
- Your program will be difficult to read and understand, because it will look like you are doing error handling when you aren't. 
- Exceptions incur quite a bit of overhead _when exceptions are thrown_, but otherwise do not usually impact performance to any significant degree. If you use exceptions to handle expected program branching you will throw many exceptions and incur a serious performance cost. 

We'll take a look now at how to do this in practice, starting with catching exceptions thrown by functions that you use. 

## Catching Exceptions

We'll start by looking at how to handle an error thrown by an existing function, such as a range error thrown by a vector. When such a function encounters an erorr and _throws_ an exception, it needs to be _caught_. 

- We first need to identify the code that could throw the exception. We do this with the `try{...}` keyword.
    - This tells our compiler that we want to monitor the execution of this code block (inside the `{}`) for exceptions. 
- We then need to intercept any exceptions which are thrown and respond to them. We do this with the `catch(){...}` keyword.

Let's take a look at an example of how to catch an exception thrown by a standard library function:

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

using std::vector;

int main()
{
    vector<int> fibbonacciList = {1, 1, 2, 3, 5, 8, 13};

    try
    {
        std::cout << fibbonacciList.at(15) << std::endl;
    }
    catch(const std::out_of_range &e)
    {
        std::cerr << "Problem accessing Fibbonacci list due to range: " << e.what() << std::endl;
        std::cerr << "Max index of list is " << fibbonacciList.size() -1 << std::endl;
    }
    catch(const std::exception &e)
    {
        std::cerr << "Other error occurred accessing Fibbonacci list: " << e.what() << std::endl;
    }

    return 0;
}
```

- We get the exception definitions from `#include <stdexcept>`.
- If we think a code block might throw an exception then we can place it inside a `try{ code block }` statement.
    - In this case we are accessing a vector, which may not be long enough. 
- After a `try{}` block we need a `catch(exception_type e){ code block }` statement. Inside the curly braces of the catch statement we put the code that we want to execute if an exception of that type is raised.
    - Some catch blocks could, for example, make corrections to values, adopt some kind of default setup, or simply log detailed error messaging and terminate the program. 
- Once the `catch` block has been executed, the program will continue as normal from after the `try`/`catch` blocks. (In this case, to `return 0;`) **It does not go back to finish the `try` block which was interrupted, so anything inside that block that occurs after the exception is thrown will remain unexecuted.** This is an important point to bear in mind for some potential errors that we will discuss later. 
- We can use `e.what()` to get the exception's message, which should report some useful information about why the exception was raised. 
- We can have multiple `catch` statements after a single `try` statement to handle different kinds of exceptions which could be thrown from the same code, for example one block for `std::out_of_range`, and another for `std::overflow_error`. 
- All exceptions inherit from `std::exception`, which can be used from the `<exception>` header, so if you want a catch block to catch generic exceptions you can write `catch(std::exception e){}`. 
- If you want to catch anything that has been thrown and you don't want to access any information in the exception itself you can also use `catch(...)`, but you will usually want to name your exception variable so that you can report information from it. 
- `catch` clauses will be evaluated in order, so you should always list your `catch` statements from most specific to most general i.e. list _derived classes_ before the _base classes_ from which they inherit. For example, `std::out_of_range` is a sub-type of `std::exception` since the `out_of_range` class inherits from `exception`. This means that: 
    - if `catch(std::exception e)` comes before `catch(std::out_of_range e)` then all `out_of_range` errors will be caught by the more general `exception` clause, and the specialised `out_of_range` error handling code will never run. 
    - if `catch(std::out_of_range)` is placed first, then the `catch(std::exception e)` code will only run for exceptions which are not `out_of_range`. 
- `cerr` is a special output stream for errors; we can use this if we want the error to be written to a different place than standard output (e.g. standard ouput to file and errors to terminal, or vice versa). We can also output exception information to `cout` though. 

We can see in this example that using `try` and `catch` blocks have significant advantages for someone reading our code:

- It is immediately obvious which part of our code we are checking for errors. 
- It is immediately obvious what is error handling code, and what is normal execution, because error handling code is all contained inside the `catch` statements. This separation of normal and exceptional behaviour can make it easier to understand what the normal operation of the code actually is and therefore what the code should do most of the time. 

As mentioned above, exceptions that aren't caught by the calling function can propagate up the call stack until they are caught (or reach the top and stop program execution). This is useful because not all errors are best handled by the function immediately above in the call stack. Consider an application that takes input from the user, and then performs many nested operations on it. At some point in this process, it may discover an issue that means that the input provided is not viable (which may not be checkable at the point of input), and throw an exception. This would most likely be handled by asking the user for a different input, which means that this is where we should place the `try ... catch` blocks. A simplified example of this kind of structure could look like this:

```cpp
#include <iostream>
#include <cmath>

using std::cout;
using std::cin;
using std::endl;

double g(double x)
{
    if(x <= 0)
    {
        throw std::logic_error("g(x) not defined for x <= 0");
    }
    return log(x);
}

double h(double x)
{
    if(x < 0)
    {
        throw std::logic_error("h(x) not defined for x < 0");
    }
    return sqrt(x);
}

double p(double x)
{
    return x*x - 3*x + 12;
}

double f(double x)
{
    return g(p(x)) * h((p(x)));
}

int main()
{
    bool complete = false;
    double x = 0;

    cout << "Enter a number." << endl;
    cin >> x;

    while(!complete) 
    {
      try 
      {
        double y = f(x);
        cout << y << endl;
        complete = true;
      } 
      catch(std::exception &e) 
      {
        cout << e.what() << endl;
        cout << "Please enter a different number." << endl;
        cin >> x;
      }
    }

    return 0;
}
```
- While this case would be trivial to check for at input, for many processes it may require actually running the code to find out if an input is acceptable or not (for example if `p(x) <= 0` were not analytically tractable). 
- Note that the exception is in this case not handled by the code that directly calls the function which throws: the exception is thrown by `g` or `h`, which is called by `f`; `f` has no catching code so the exception propagates up another layer in the call stack to `main`. Here we finally catch and handle it by reporting the problem and asking the user for a new input. 

## Throwing Exceptions

We can also throw exceptions from our own functions, which allows code which calls our functions to handle errors that might occur within our function code. For example, we might have a function which calculates the inner product of two vectors. In this case, the number of elements in both vectors must be the same for this to make sense! So we could write code like this:

```cpp
double InnerProduct(const vector<double> &x, const vector<double> &y)
{
    if(x.size() != y.size())
    {
        std::string errorMessage = "Inner product vectors different sizes: " + std::to_string(x.size()) + " and " + std::to_string(y.size()); 
        throw std::range_error(errorMessage);
    }
    
    double product = 0;
    for(size_t i = 0; i < x.size(); i++)
    {
        product += x[i] * y[i];
    }

    return product;
}
```

This will throw a `range_error` if the two vectors are not the same size. The code which calls this function can catch this exception using `catch(std::range_error e){}`. 

```cpp
int main()
{
    vector<double> a = {0.2, 0.1, 1.2, 5.99};
    vector<double> b = {0.1, 1.8, 2.9};

    double ab;
    try
    {
        ab = InnerProduct(a, b);
    }
    catch(const std::range_error& e)
    {
        std::cerr << e.what() << '\n';
    }
    

    return 0;
}
```

**Warning** You can actually throw _any_ type in C++, it doesn't have to inherit from `exception`. The compiler will not complain if you throw, for example, an `int` or a `string`. To do so is bad practice, as these objects are not designed to be used to carry error information and codes which may call your functions will not be expecting them. This means calling code is unlikely to check for them, which will allow them to pass to the top level uncaught and halt execution. Restricting yourself to throwing classes which inherit from `exception` will make your code easier to understand for others, and compatible with other code bases. If you need values of other types to be reported with your exception then you can include them as member variables in you own exception class (see below). 

## Defining Our Own Exceptions 

We've mentioned above that we can differentiate between different kinds of exceptions by checking for different expception classes, and then execute different error handling code accordingly. This is a very powerful feature of exceptions that we can extend further by defining our own exception classes to represent cases specific to our own applications. When we define our own exceptions, they should inherit from the `std::exception` class, or from another class which derives from `std::exception` like the standard library exceptions listed above. You should be aware though that if you inherit from a class, for example `runtime_error`, then your exception will be caught by any `catch` statements that catch exceptions of the base classes (`runtime_error` or `exception`). 

Exceptions that we define should be indicative of the kind of error that occur. Rather than trying to create a different exception for each function that can go wrong, create exception classes that represent kinds of problems, and these exceptions may be thrown by many functions. When creating new exception classes it is a good idea to think about what is useful for you to be able to differentiate between. 

- For example, an arithmetic overflow error is a useful class of errors which can tell us that our values have become too large to be handled. 
- We don't want a separate arithmetic error class for every numerical function that we write that could go wrong!

When we inherit from the `std::exception` class we inherit a virtual function called `what()`, which can be used to read or print out error messages associated with that exception. 

To override `what()` the type declaration is:

```cpp
const char * what() const noexcept {...}
```

- The return type is `const char *` i.e. a constant `char` pointer. This is a C-style array of characters, and is how strings are handled in C. 
- The `const` after the function name enforces that no member variables can be changed inside the function body i.e. `what()` cannot change any of the exception's data.
    - You can mark special variables to be modifiable even in `const` functions, by declaring them `mutable` e.g. `mutable int x`. Usually this is a practice that is best avoided as it makes your code more difficult to reason about. 

Derived classes `runtime_error`, `logic_error`, and `failure` all contain constructors which take arguments of type `const string &` (reference to a constant string), which sets the error message returned by `what()`. These can be useful if you want to be able to set the message without overriding the `what()` function. 

Bear in mind however that you can also add any functionality that would be useful for your exception class, such as additional member variables which store relevant data. 

Here's an example of an exception class that overrides the `what()` method and also returns special data. 

```cpp
class FunctionDomainException: public exception
{
    public:
    FunctionDomainException(string func_name, double value) 
    {
        message = "Function Domain error on function " + func_name \
                 + ". Input " + std::to_string(x) + " invalid.";
        bad_input = x;
    }

    const char * what() const noexcept
    {
        return message.c_str();
    }

    string message;
    double bad_input;
};
```

- The constructor takes a `string` (`func_name`) to report the name of the function and a `double` (`value`) to report the value that the function failed at. 
- It then constructs an error message based on this information and also stores `value` value as a member variable `bad_input`. 
- The `what()` method is overridden to print out the string that we've constructed. 
- If we catch this error we can also access `bad_input` as it is public, which may be useful for us to be able to manipulate in numerical code rather than just printing it out. 

## Control flow and memory management 

We've discussed above that raising an exception will prematurely halt the execution of a function and return control to the calling function. It will also halt the execution of any calling functions until we find ourselves within a `try` block, at which point the `catch` code is executed. We should always be aware of what our program will not do if an exception is thrown. 

- If you place a try block around a function which may throw exceptions at multiple points (it may have multiple `throw` statements or make calls to a number of other functions which could themselves throw exceptions) and you are passing variables in by reference to be modified, you should be aware of the possible states that your data could be if the function is prematurely halted. Not all the changes that your function is intended to make on your data may have happened!
- When we reach the catch block, the stack memory for any functions which threw exceptions and were halted is freed (since they are now out of scope). This means that stack variables are cleaned up, and destructors for any stack variables are called (including smart pointers, the destructors for which de-allocate the data to which they point). Be aware though that if there are _raw pointers_ on the stack, the memory that they point to is not deleted (only the pointer itself is) and so if the memory that it points to is not a stack variable or also owned by a smart pointer a memory leak will occur. **This is one of the reasons why we should not use raw pointers for memory ownership.** If you do have an owning raw pointer in a function and you want to throw and exception, it is vital that you use `delete` to free the memory before throwing the exception; likewise you must be aware of any function calls that you make which could themselves throw an exception: these _must_ be caught so that you can free the memory before returning control to the call stack. 