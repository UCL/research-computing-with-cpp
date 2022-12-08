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

You may notice these exceptions being thrown by a number of different methods that you use in C++! For example, if you try to access a `vector` using `.at(i)` outside of the range of the vector, an `out_of_range` exception will be thrown, which will halt program execution and you can see it reported the terminal output if you don't handle it properly. 

When you look up a function in the C++ documentation, you can see what exceptions it can throw, and therefore what you may need to check for if necessary. Some functions have "no throw" guarantees, which means that they cannot throw exceptions. Just because a function does not throw exceptions does not mean it is impossible for an error to occur: be sure to check if the function is using some other method of reporting and handling errors. 

## The basic idea of Exceptions

An exception is a type of object which can be "thrown" when something happens which is not supposed to happen, but which has been anticipated by the programmer. For example, if you write a function which requires the use of the first three elements of a vector, your function should check that the vector passed to it has at least three elements. One way of handling the case where it does not have enough elements is to throw an exception: this halts the execution of the function (thereby preventing any attempts to, for example, access elements that aren't there), and returns to execution to the calling code. If the code which called the function has been set up to detect exceptions, then it can "catch" the exception and handle it appropriately. The the exception is not caught by the calling function, then that function halts as well and the exception propagates up the stack to that function's caller, and so on until it is caught. Exceptions which are thrown but not caught by any enclosing code will cause the program to terminate prematurely.

**N.B.** Exceptions should not be used a method of controlling program flow (like, for example, `if ... else` statements) and should only be used to cover unusual cases. **Do not use throw exceptions as an alternative way of returning a value from a function or as a way to exit a loop.** These should always be handled using `return` statements (for function return) and `break` statements (for loop exit). Exception handling statements should be used to handle cases where normal execution of the program simply cannot continue, and in a typical execution of the code the exception should not get raised. Some good examples of where to use exceptions for error handling are: running out of space for a process that needs to allocate memory, file system input/output errors, runtime errors from users giving unexpected bad input. Using exceptions to control program flow other than error handling can be a problem for two reasons:
- You program will be difficult to read and understand, because it will look like you are doing error handling when you aren't. 
- Exceptions incur quite a bit of overhead _when exceptions are thrown_, but otherwise do not usually impact performance to a large degree. If you use exceptions to handle expected program branching you will throw many exceptions and incur a serious performance cost. 

We'll take a look now at how to do this in practice, starting with catching exceptions thrown by functions that you use. 

## Catching Exceptions

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
- If we think a code block might throw an exception that we can place it inside a `try{ code block }` statement.
- After a `try{}` block we need a `catch(exception_type e){}` block. Inside the curly braces of the catch block we put the code that we want to execute if there an exception is raised. This means that we don't have to halt execution when the exception is raised any more! 
- We can use `e.what()` to get the exception's message, which should report some useful information about why the exception was raised. 
- We can have multiple `catch` statements after one `try` block to handle different kinds of exceptions, for example one block for `std::out_of_range`, and another for `std::overflow_error`. 
- All exceptions inherit from `std::exception`, which can be used from the `<exception>` header, so if you want a catch block to catch generic exceptions you can write `catch(std::exception e){}`. 
- If you want to catch anything that has been thrown and you don't want to access any information in the exception itself you can also use `catch(...)`, but you will usually want to name your exception variable so that you can report information from it. 
- `catch` clauses will be evaluated in order, so you should always list your `catch` statements from most specific to most general. For example, `std::out_of_range` is a sub-type of `std::exception` since the `out_of_range` class inherits from `exception`. This means that is `catch(std::exception e)` comes before `catch(std::out_of_range e)` then all `out_of_range` errors will be caught by the more general `exception` clause, and the specialised `out_of_range` error handling code will never run. On the other hand, if `catch(std::out_of_range)` is placed first, then the `catch(std::exception e)` code will only run for exceptions which are not `out_of_range`. 
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
        throw std::logic_error("g(x) not defined for x < 0");
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

## Throwing Exceptions

We can also throw exceptions from our own functions, which allows code which calls our functions to handle errors that might occur within our function code. For example, we might have a function which calculates the inner product of two vectors. In this case, the length of both vectors must be the same for this to make sense! So we could write code like this:
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
This will throw a `range_error` if the two vectors are not the same size. The code which calls this function can catch this exception using `catch(std::range_error e)`{}. 
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

**Warning** You can actually throw _any_ type in C++, it doesn't have to inherit from `exception`. The compiler will not complain if you throw, for example, an `int` or a `string`. To do so is bad practice, as these objects are not designed to be used to carry error information and codes which may call your functions will not be expecting them. This means calling code is unlikely to check for them, which will allow them to pass to the top level uncaught and halt execution. Restricting yourself to throwing classes which inherit from `exception` will make your code easier to understand for others, and compatible with other code.  

## Control flow and memory management 

We've discussed above that raising an exception will prematurely halt the execution of a function and return control to the calling function. It will also halt the execution of any calling functions until we find ourselves within a `try` block, at which point the `catch` code is executed. We should always be aware of what our program has not done if an exception is thrown. 
- When we reach the catch block, the stack memory for any calling functions which were halted is freed. This means that stack variables are cleaned up, and destructors for any stack variables are called. Be aware though that if there are _raw pointers_ on the stack, the memory that they point to is not deleted (only the pointer itself is) and so if the memory that it points to is not a stack variable or also owned by a smart pointer a memory leak will occur. **This is one of the reasons why we want to avoid using raw pointers for memory ownership.** If you do have an owning raw pointer in a function and you want to throw and exception, it is vital that you use `delete` to free the memory before throwing the exception; likewise you must be aware of any function calls that you make which could themselves throw an exception: these _must_ be caught so that you can free the memory before returning control to the call stack. 
- If you place a try block around a function which may throw exceptions at multiple points (it may have multiple `throw` statements or make calls to a number of other functions which could themselves throw exceptions) and you are passing variables in by reference to be modified, you should be aware of the possible states that your data could be if the function is prematurely halted. 

## Defining Our Own Exceptions 

We've seen above that we can differentiate between different kinds of exceptions by checking for different expception classes, and then execute different error handling code accordingly. This is a very powerful feature of exceptions that we can extend further by defining our own exception classes to represent cases specific to our own applications. When we define our own exceptions, they should inherit from the `std::exception` class, or from another class which derives from `std::exception` like the standard library exceptions listed above. You should be aware though that if you inherit from a class, for example `runtime_error`, then your exception will be caught by any `catch` statements that catch `runtime_error` types. 

Exceptions that we define should be indicative of the kind of error that occur. Rather than trying to create a different exception for each function that can go wrong, create exception classes that represent kinds of problems, and these exceptions may be thrown by many functions. When creating new exception classes it is a good idea to think about what is useful for you to be able to differentiate between. 

When you write an exception inheriting from `std::exception`, you will inherit the constructor and the `.what()` method from the base class, so you will be able to use it as an ordinary exception (see next week when we discuss polymorphism). Bear in mind though that you can also extend this functionality, and add additional methods or data field if this will help your error handling code.  