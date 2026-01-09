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
4. C++ has strong support for object-oriented programming, including a number of features not present in Python. These features allow us to create programs that are safer and more correct, by allowing us to define objects that are guaranteed to maintain particular properties (called _invariants_). For example, defining a kind of list that is always sorted, and can't be changed into an un-sorted state, means that we can use faster algorithms that rely on sorted data _without having to check that the data is sorted_. 
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
- Compilation and program structure means there's a bit of overhead to starting a C++ project, and you can't run it interactively. This makes it harder to jump into experimenting and plotting things the way you can in the Python terminal or notebooks.
- C++ is less well known in more general research communities, so isn't always the most accessible choice for collaboration outside of HPC. (You can also consider creating Python bindings to C or C++ code if you need the performance but your collaborators don't want to deal with the language!)

For larger scale scientific projects where performance and correctness are critical, then C++ can be a great choice. This makes C++ an excellent choice for numerical simulations, low-level libraries, machine learning backends, system utilities, browsers, high-performance games, operating systems, embedded system software, renderers, audio workstations, etc, but a poor choice for simple scripts, small data analyses, frontend development, etc. If you want to do some scripting, or a bit of basic data processing and plotting, then it's probably not the best way to go (this is where Python shines). For interactive applications with GUIs other languages, like C# or Java, are often more desirable (although C++ has some options for this too). 

I'd also like to emphasise that while we _use_ C++, the goal of this course is not to simply teach you how to write C++. **This is a course on software engineering in science, and the lessons should be transferable to other languages.** Languages will differ in the features and control that they offer you, but understanding how to write well structured, efficient, and safe programs should inform all the programming that you do. 
 

# Writing in C++: Hello World!

Here's a little snippet of C++:

```cpp
#include <iostream>

using std::cout;

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

Now we can actually *run* the program with `./hello_world`. You should see output similar to:

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
using std::cout;
```

Classes (types of object) and functions from the standard library are prefaced with `std::`. This is called a _namespace_; it's used to avoid name clashes in large programs, and the standard library especially has a large amount in it with common names like `vector`, `map`, and `array` that could easily be used in other ways by other libraries or parts of your program! If you want to avoid writing `std::` all the time, you can use the `using` to make the class or function available without namespacing. You can also import an entire namespace, for example:

```cpp
using namespace std;
```

This can avoid writing lists of `using` statements but runs the risk of name clashes. It's not usually good practice to import the entire `std` namespace because the potential for clashes is high, although it can be fine for small programs. 

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

The line `int main()` is a *function signature*, and it is being used to begin a *function definition* (which is a function signature combined with the "body" which defines what the function actually does). Function signatures have three mandatory parts, presented in this order:

- Return type; this can be `void` if the function returns nothing.
- Function name.
- Function arguments; these go inside the brackets `()` and must include the type of each argument. A function which takes no arguments can have empty brackets.

Function signatures can be more complex than this as we shall see later on, but these are the components that must always be present. 

Here we're describing a function called `main` that returns a single value of type `int`, and takes no parameters or arguments, which we know because the brackets `()` are empty. A C++ program that is intended to be compiled into an executable (as opposed to a library; more on that later) must contain a function called `main`: this is the entry-point to our program. **`main` is the function which is executed when the program starts.**

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





