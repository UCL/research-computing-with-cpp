---
title: Introduction to C++
---

# Week 1: Introduction to C++

## What is C++?

C++ is a relatively low-level*, relatively old, general-purpose programming language, commonly used for programs that require high performance. First released in 1985 by Bjarne Stroustrup, it was intended as a successor to C, primarily to add object-oriented features. Now, C++ is multi-paradigm, meaning that it includes features that enable object-oriented, functional, and procedural programming. It inherits many characteristics from C: it is compiled, has a static type system, and allows manual memory management. All these features ultimately enable C++ to perform extremely well; these same features can be tricky to use effectively and can be a source of bugs. It is the balance between programmer productivity and performance that allows C++ to be used to build large, complex, fast applications and libraries. 

Since 1998, C++'s development has been governed by an ISO working group that collates and develops new C++ features into a *standard*. Since 2011, a new standard has been released every 3 years, with compiler support for new features lagging behind by up to 3 years.

This course will assume the use of the `C++17` standard, which is presently widely supported by compilers. (Compiler support tends to lag behind the release of the C++ standard, since the compiler developers need time to implement it and check that the new compilers work properly!)

> \* The exact meaning of "low level" depends on whom you ask, but in general _lower level_ languages are languages that more directly represent the way that the machine works, or give you control over more aspects of the machine. By contrast, _higher level_ languages abstract more of that away and focus on defining the outcome that you want. Originally, a "low level" language refers to _assembly code_, which is where the instructions that you write are (a human readable version of) the instruction set of the machine itself! This code is by its nature specific to a given kind of machine architecture and therefore not portable between systems, and doesn't express things in a way that is intuitive to most people. High level languages were introduced to make code easier to understand and more independent of the hardware; the highest level languages, like Haskell, are highly mathematical in their structure and give hardly any indication of how the computer works at all! C++ falls somewhere in the middle, with plenty of high level abstractions and portability, but it still gives us some features associated with low level programming like direct addressing of memory. This extra degree of control is very valuable when you need to get the best out of systems that require high performance or have limited resources.

## Why are we using C++?

The most common language for students to learn at present is probably Python, and many of you may have taken the Python based software engineering course last term. So why are we now changing to C++?

1. C++ is the standard language for high performance computing, both in research and industry. 
2. C++ code runs much faster than native Python code. Those fast running Python libraries are written in C! As scientific programmers, we sometimes have to implement our own novel methods, which need to run efficiently. We can't always rely on someone else having implemented the tools that we need for us. 
3. C++ is a great tool for starting to understand memory management better.
    - Most code that we write will not need us to allocate and free resources manually, but C++ gives us a clear understanding of when resources are allocated and freed, and this is important for writing effective and safe programs. 
    - Many structures in C++ have easy to understand and well defined layouts in memory. The way that data is laid out in memory can have a major impact on performance as we shall see later, and interacting with some high performance libraries requires directly referencing contiguous blocks of memory. 
4. C++ has strong support for object-oriented programming, including a number of features not present in Python. These features allow us to create programs that are safer and more correct, by allowing us to define objects which have properties that have particular properties (called _invariants_). For example, defining a kind of list that is always sorted, and can't be changed into an un-sorted state, means that we can use faster algorithms that rely on sorted data _without having to check that the data is sorted_. 
5. C++ is multi-paradigm and gives us a lot of freedom in how we write our programs, making it a great language to explore styles and different programming patterns. 
6. C++ has a static type system (as do many other languages), which is quite a big shift from Python's dynamic typing. Familiarity with this kind of type system is extremely useful if you haven't used it before, and as we will see it can help us to write faster code with fewer bugs.

### Why is C++ fast?

Because a C++ program is compiled before the program runs, it can be much faster than interpreted languages. Not only is the program compiled to native machine code, the lowest-level representation of a program possible with today's CPUs, compilers are capable of performing clever optimisations to vastly improve runtimes. With C++, C, Rust, Fortran, and other natively compiled languages, there is no virtual machine (like in Java) or interpreter (like in Python) that could introduce overheads that affect performance.

Many languages use a process called _garbage collection_ to free memory resources, which adds run-time overheads and is less predictable than C++'s memory management system. In C++ we know when resources will be allocated and freed, and we can run with less computational overhead, at the cost of having to be careful to free any resources that we manually allocate. (Manually allocating memory is relatively rare in modern C++ practices! This is more common in legacy code or C code, with which you will sometimes need to interact.)

Static type checking also helps to improve performance, because it means that the types of variables do not need to be checked during run-time, and that extra type data doesn't need to be stored.

### Should I just use C++ for everything? 

Probably not! 

Choosing a programming language is a mixture of picking the right tool for the job, and the right tool for you (or your team). Consider some of the pros and cons for a given project you are working on.

C++ Pros:
- Can produce code which is both fast and memory efficient. 
- Very flexible multi-paradigm language supports a lot of different approaches. 
- Gives you direct control of memory if you need it. 
- Large ecosystem of libraries. 
- Can write code which runs on exciting and powerful hardware like supercomputing clusters, GPUs, FPGAs, and more!
- Can program for "bare metal", i.e. architectures with no operating system, making it appropriate for extremely high performance or restrictive environments such as embedded systems.
- Static typing makes programs safer and easier to reason about. 
- C++ is well known in high performance computing (HPC) communities, which is useful for collaborative work.

C++ Cons:
- Code can be more verbose than a language like Python.
- C++ is a very large language, so there can be a lot to learn.
- More control also means more responsibility: it's very possible to cause memory leaks or undefined behaviour if you misuse C++.
- Compilation and program structure means there's a bit of overhead to starting a C++ project, and you can't run it interactively. This makes it harder to jump into experimenting and plotting things the way you can in the Python terminal. 
- C++ is less well known in more general research communities, so isn't always the most accessible choice for collaboration outside of HPC. (You can also consider creating Python bindings to C or C++ code if you need the performance but your collaborators don't want to deal with the language!)

For larger scale scientific projects where performance and correctness are critical, then C++ can be a great choice. This makes C++ an excellent choice for numerical simulations, low-level libraries, machine learning backends, system utilities, browsers, high-performance games, operating systems, embedded system software, renderers, audio workstations, etc, but a poor choice for simple scripts, small data analyses, frontend development, etc. If you want to do some scripting, or a bit of basic data processing and plotting, then it's probably not the best way to go (this is where Python shines). For interactive applications with GUIs other languages, like C# or Java, are often more desirable (although C++ has some options for this too). 

I'd also like to emphasise that while we _use_ C++, the goal of this course is not to simply teach you how to write C++. **This is a course on software engineering in science, and the lessons should be transferable to other languages.** Languages will differ in the features and control that they offer you, but understanding how to write well structured, efficient, and safe programs should inform all the programming that you do. 
 

# Writing in C++: Hello World!

Here's a little snippet of C++:

```cpp
#include <iostream>

using namespace std;

int main() 
{
    cout << "Hello World!\n";
    
    return 0;
}
```

We'll go through exactly what each line and symbol mean in a moment, but let's first run this program. Since C++ is a *compiled* language, we must use a compiler to turn this code into an executable that the computer can run. In this course we'll nearly always be using the GNU C++ compiler, `g++`. We can compile the above code pasted into a file called `main.cpp` into an executable called `hello_world` by running in the terminal:

```bash
g++ main.cpp -o hello_world
```

> `.cpp` is the standard suffix for C++ files. You will also encounter `.h` and `.hpp` suffixes, which describe *header files* (we'll come to those later) and sometimes `.c`, describing just a C file.

Now we can actually *run* the program with `./hello_world`[^dot_slash]. You should see output similar to:

```
[my_username@my_hostname my_current_folder]$ ./hello_world
Hello World!
```

> The `./` syntax before the executable name is required because when a command (or executable name) is entered in Linux, it does not automatically search the current directory for the entered command. We need to tell the Linux shell that the command is inside the current directory, which is labelled `.`, hence the full command `./hello_world` is telling the Linux shell to "run the executable `hello_world` in the current directory".

Congratulations, you have just run your first C++ program!

## Deconstructing Hello World

At the highest level, this C++ code has 3 parts, an include statement, a using statement and a single function called `main`. Let's go through each piece individually.

**The include statement**

The first line we see in Hello World is an *include statement*:

```cpp
#include <iostream>
```

We'll go into what *exactly* this line is doing later but all you need to know at this stage is that include statements are how we import external code into the current file so we can use the included functions, classes, etc in our code. This is similar to `import` in Python, Java and Matlab. In the specific line above we're including the code from `iostream`, the piece of the C++ standard library that provides input-output functionality. We require this to use `cout` later. The *hash* or *pound* symbol `#` and the angle brackets `<...>` will be explained later when we get into the *preprocessor*.

**The using statement**

The next line is a *using statement*:

```cpp
using namespace std;
```

Again, we'll explore what this whole line actually does later but the short version is it allows us to access functions and classes inside the standard library without typing `std::` everywhere. I recommend you use it but **only in .cpp files**.

There's also an interesting bit of punctuation here; you have probably noticed the line ends in a semicolon `;`. C++, and many other languages, require this because the language doesn't care about (most) *whitespace*. Technically we can write our whole program all on one line, ignoring all indentation and newlines:

```cpp
#include <iostream> using namespace std; int main() {cout << "Hello World!\n"; return 0;}
```

But this looks horrible so we use newlines and indentation in appropriate places to make our code more *readable*.

This is very different to languages where whitespace has meaning, like Python, where a *newline* usually denotes the end of a _statement_, and indentation denotes a new code block or scope. Because of this, C++ requires all *statements* to end with a semicolon so that the compiler knows exactly where a line is meant to end. In practice this means nearly every line will end with a semicolon, except lines starting with `#` (we'll come back to that) and lines that open or close *scopes* with the curly braces `{}`.

**Functions**

Here's our `main` function again:

```cpp
int main() 
{
  ...
  return 0;
}
```

This is made up of three main parts:

1. The *function signature*, `int main()`
2. The *function body*, the code inside the curly braces `{}`, and
3. The *return statement* inside the function body.

We'll go through each of these pieces.

**The function signature**

The line `int main() {` begins a *function definition* and includes the *function signature* (the piece without the curly brace). Here we're describing a function called `main` that returns a single value of type `int`, and takes no parameters or arguments, which we know because the brackets `()` are empty. A C++ program that is intended to be compiled into an executable (as opposed to a library; more on that later) must contain a function called `main`: this is the entry-point to our program. **`main` is the function which is executed when the program starts.**

The signatures of other functions might look like:

A function called `to_string` that takes an `int` called `x` and returns a `string`:

```cpp
string to_string(int x)
```

A function called `is_larger` that takes two `float`s and returns a `bool`:

```cpp
bool is_larger(float a, float b)
```

A function called `save_to_file` that takes a `vector` (a kind of array) of `int`s  and has *no return value*, which we denote with `void`:

```cpp
void save_to_file(vector<int> my_data)
```

**The function body**

The function body is simply all code within the curly braces. This is the code which defines what the function actually does, and what value(s) (if any) it returns. Let's look at the statement that prints "Hello World!":

```cpp
cout << "Hello World!\n";
```

First, `cout` is an example of what is called an *output stream*; `cout` will normally write to your terminal. (Other kinds of streams exist, including for reading and writing files.) In essence, `cout << string` is equivalent to `print(string)` in Python, although there are some nuances that will be discussed when we discuss *operators* like `<<` in more detail. For now, all you need to know is `cout << ...` is how we print to the standard output in C++.

You'll notice the string that we want to print is surrounded by double-quotes `"`. These are used to write *string literals* in code, while single-quotes `'` are used to write *character literals* like `'a'`, `'/'` or `'@'`. When inside the string literal, we use the special *newline character* `'\n'` to make the program print a newline. Although we type this as two characters, a `\` and an `n`, C++ interprets this as a character with special meaning. See [Escape Characters](https://en.wikipedia.org/wiki/Escape_character) for more information.

**The return statement**

In our main function, the return statement is:
```cpp
return 0;
```
This statement is required in every function, except functions with a `void` return type (however even then a function can use `return;` without a value to end the function early). If we reach a `return` statement our function will terminate and our program will continue executing from the place where we called our function, even if there is more code after the `return` statement. 

This particular return statement is inside our `main` function which expects to return an `int` to its calling code. Since this is the return value from the special function `main`, this value is interpreted by the operating system as a *status code*, a number between 0 and 255 that gives some information about the success or failure of the application. 

Like most common languages, functions are called like:
```cpp
int main() 
{
  string x_as_string = to_string(3);
}
```
where the *return value* is, here, assigned to the variable `x_as_string`. We can call functions without using the return value by just not assigning the function call:
```cpp
int main() 
{
  to_string(3);
}
```

**Types**

We've already come across, `int`, `float`, `bool` and other keywords that define the *type* of variables and parameters. C++ is a *strongly-typed* and *statically-typed* language, so the *types* of every single variable or function parameter must be known at *compile time*, and those types cannot change during runtime. 

> **Compile time** refers to the time at which the code is compiled. It's used in contrast to **runtime** or **run time** which refers to the time at which the program is run. For example, if a program requires a number to do some computation and you can write that number in the source code itself, that number is known at *compile time*. If, say, the user needs to input that number, the value is only known at *run time*.

This is in contrast to the *weakly-typed* and *dynamically-typed* Python where we can define a variable `x` and assign it the integer value `2`, then reassign a string `"2"` to the same variable!

```python
x = 2
x = "2"
```

In C++ this kind of code will produce an error because, to reiterate:

1. We must give all variables a type, and
2. We cannot change the type of a variable

Again, we'll explore later what types are available in C++ (and how we can create our own) but a useful initial list is:

- `bool`: a boolean value, i.e. `true` or `false`
- `int`: an integer value, e.g. `-4, 0, 100`
- `float`: a 32-bit floating-point value `-0.2, 0.0, 1.222, 2e-3`
- `double`: a 64-bit floating-point value (same as `float` but can represent a greater range and precision of real numbers)
- `char`: a single character, e.g. `'a', 'l', ';'`
- `string`: a kind of list of characters, used to represent text. 
  - Not to be confused with a *character array* which can be difficult to deal with.
- `std::vector<T>`: a kind of array of elements of type `T`, e.g. `std::vector<int> {1, 100, -1}` declares a vector of integers.
  - Not to be confused with a *mathematical* vector, this is similar to a Python `list`.
  - `std::` means that the vector type is part of the **C++ Standard Library** namespace. If you have the line `using namepsace std;` in your code then you don't need to use the `std::` prefix after that. Just bear in mind that this could cause name clashes if you delare another type or function which has the same name as something in the standard library!
  - **N.B.** you need to add `#include <vector>` to the top of your program to use the vector type. 

## Basic control structures

C++ contains many of the same control structures as other programming languages that you have used previously. 

### Conditional logic using `if`/`else` statements

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

### Loops (`for` and `while`)

```cpp=
for(uint i = 0; i < 100; ++i)
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
- `uint` is a type for _unsigned integers_, which are integers that cannot be negative. It's a good idea to use these for counting and other values which shouldn't be less than 0. 
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

## Programs With Multiple Files

Like other programming languages, it is possible (and good practice!) to break up C++ programs into multiple files. C and C++ however have a slightly unusual approach to this compared to some other languages. Let's consider that we have two C++ files, our `main.cpp` which contains our `main` function (the entry point for execution of our program), and another which defines some function that we want to use in `main`. 

**main.cpp:**
```cpp=
#include <iostream>

int main()
{
    // Call function f
    int x = f(5, 2);
    
    std::cout << "x = " << x << std::endl; // endl stands for end line
    
    return 0;
}
```

**function.cpp:**
```cpp=
int f(int a, int b)
{
    return (a+2) * (b-3);
}
```

We could try to just compile these files together:
```
g++ -o ftest main.cpp function.cpp
```
but this won't work! 
In order to compile `main.cpp`, we need to know some information about the function `f` that we are calling. 

Let's take a moment to understand C++'s compilation process a little better.
- C++ is designed so that different source files can be compiled **independently, in parallel**.
    -  This helps to keep compilation times down. 
    -  It also means that if we only change one part of the program, we only need to re-compile that part. The rest of the program doesn't need to be compiled again!
- Remember though that C++ is _statically typed_, which means **the compiler needs to know the types of all variables and functions used in the code it is compiling, at compile time**. Otherwise it cannot check that the code you have written is correctly typed!
    - Take for example the statement `int x = f(5, 2);` in `main.cpp`. In order for this to be correctly typed, we need to know that `f` can accept two numbers as its arguments, and it must return an integer, because `x` is declared to be an `int`. If we don't know the type of `f`, we can't be sure that this is true! 

Let's use this simple example program to explore how the compiler deals with our code, and what information it needs to do its job. 

### Code Order and Declarations

We'll start with a single file and work towards a multiple file version. Consider the following two versions of the same program:

**Version 1**
```cpp=
#include <iostream>

int f(int a, int b)
{
    return (a + 2) * (b - 3);
}

int main()
{
    int x = f(5, 2);

    std::cout << "x = " << x << std::endl;

    return 0;
}

```

**Version 2**
```cpp=
#include <iostream>

int main()
{
    int x = f(5, 2);

    std::cout << "x = " << x << std::endl;

    return 0;
}

int f(int a, int b)
{
    return (a + 2) * (b - 3);
}
```

Only the first of these two programs will compile! 
- **C++ will parse your file in order**, and so in the second version it comes across the function `f` _before_ it has been defined. The compiler doesn't know what to do! It can't know what `f` is supposed to be, and if this is a valid & type-safe statement. 
- C++ does **not** need to know everything about `f` ahead of time though; it just need to know _what_ it is and what its type is. This is the job of **forward declaration**: something that tells us what the type of the symbol is without telling us exactly what it does. In this case, this would be a **function declaration**, but we can also have declarations for other things in C++, as we shall see later on in the course. 

**With a function declaration:**
```cpp=
#include <iostream>

// Function declaration for f
int f(int a, int b);

int main()
{
    int x = f(5, 2);

    std::cout << "x = " << x << std::endl;

    return 0;
}

int f(int a, int b)
{
    return (a + 2) * (b - 3);
}
```

- Line 4 is the **function declaration**.
    - This defines the name `f` as a function that will be used in this program. 
    - It tells us that `f` takes two `int` arguments, and returns an `int`.
    - It does not define _what_ `f` will do or how its output is calculated. That can happen later!
    - It's worth knowing that the names `a` and `b` aren't required in a declaration; since we're not defining the behaviour here, we don't need to be able to refer to the arguments individually. `int f(int, int);` is an equally valid function declaration. Nevertheless, we usually include argument names in declarations because it makes them easier to understand and use, especially if the arguments have informative names! 
- Line 15-18 is the **function definition**.
    - This contains the actual code which is executed when the function is called. The compiler doesn't need to know how the function `f` works in order to compile `main` because it knows that the types are correct, but in order to finish building the program it will need a definition for `f`! 

This program _will_ compile, because the compiler knows when it reaches main that `f` is a symbol which stands for a function which takes two `int`s and returns an `int`. This means that it can deduce that `int x = f(5, 2);` is a valid statement. When it reachs the definition on `f` at line 15 it is then able to create the code for that function. 

This might seem like a rather pointless thing to do in a program as trivial as this, but it's a very important step towards writing programs in multiple files. 

Now that we know that we can write function declarations, we can move the function definition to a different file, and compile both files separately. 

**main.cpp**:
```cpp=
#include <iostream>

int f(int a, int b);

int main()
{
    int x = f(5, 2);

    std::cout << "x = " << x << std::endl;

    return 0;
}
```

**function.cpp**:
```cpp=
int f(int a, int b)
{
    return (a + 2) * (b - 3);
}
```

To compile these files separately we can use the `-c` flag:
```
g++ -c main.cpp
```
will compile the code for `main` into a `.o` file (`main.o`), known as an **object file**. We can compile `function.cpp` into an object file, `function.o`, in the same way. 
```
g++ -c function.cpp
```
- **Object files** are code which has been compiled but which only form partial programs. We can't execute `main.o` because the definition of `f` is missing from `main.cpp`! The program wouldn't know what to do when it reaches `f`. 
- In order to create an executable which we can run, we need to **link** the object files, using the **linker**. This gives the compiler the definition of all the functions it needs, so then it can create the machine commands to jump to the executable code for `f` whenever it is called, and jump back when it has finished. 

```
g++ -o test_f main.o function.o
```
This command will produce an executable, `test_f`, by linking the two object files `main.o` and `function.o`. In the final compiled executable, the code from `function.o` is run when `f` is called in `main`. 

For a simple project like this, we can compile an executable in one step by providing both source files to the compiler at the same time:
```
g++ -o test_f main.cpp function.cpp
```

### Header Files

Forward declarations for functions are helpful, but they can still clutter up our code if we are making use of large numbers of functions! Instead, we put these declarations in **header files**, which usually end in `.h` or `.hpp`. We use `#include` to add header files to a `.cpp` file: this allows the file to get the declaration from the header file. The definitions are not kept in the header file, they are in a separate `.cpp` file. 

In this case the files look as follows:

**function.h**:
```cpp=
int f(int a, int b);  // function declaration
```

**function.cpp**:
```cpp=
int f(int a, int b)
{
    return (a + 2) * (b - 3);
}
```

**main.cpp**:
```cpp=
#include <iostream>
#include "function.h"  // include our header file with the declaration

int main()
{
    int x = f(5, 2);
    
    std::cout << "x = " << x << std::endl;
    
    return 0;
}
```

You can compile as before, if your include file is in the same folder:
```
g++ -o test_f main.cpp function.cpp
```

If your include file is in a different folder, your need to tell the compiler where to find it using the `-I` option:
```
g++ -o test_f main.cpp function.cpp -Iinclude_folder/
```



## Useful References

- I'd highly recommend Bjarne Stroustrup's _A Tour of C++_. This comes in many different editions, covering different standards of the language, so try to use one from `C++17` onwards! This is available online from the UCL library services. 
- The [C++ core guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) are a fantastic resource for learning more about writing good quality C++ code, and some of the nitty gritty details of the language that can easily trip people up. It's rather dense, so it's best to use this to search for answers to questions you already have than just trying to read it through!! 
- The [Google C++ style guide](https://google.github.io/styleguide/cppguide.html) is also interesting, and offers good explanations for their choices. If you do use this as a resource, **don't simply adopt their practices, but read their reasons first!** Some of their reasons will absolutely not apply to you and the projects that you work on, so make sure that you're always making informed choices. 
