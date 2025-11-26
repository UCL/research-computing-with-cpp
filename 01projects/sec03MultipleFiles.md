---
title: C++ Programs with Multiple Files
---

# Programs With Multiple Files

Like other programming languages, it is possible (and good practice!) to break up C++ programs into multiple files. C and C++ however have a slightly unusual approach to this compared to some other languages. Let's consider that we have two C++ files, our `main.cpp` which contains our `main` function (the entry point for execution of our program), and another which defines some function that we want to use in `main`. 

**main.cpp:**
```cpp
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
```cpp
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

## Code Order and Declarations

We'll start with a single file and work towards a multiple file version. Consider the following two versions of the same program:

**Version 1**
```cpp
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
```cpp
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
- C++ does **not** need to know everything about `f` ahead of time though; it just need to know _what_ it is and what its type is. This is the job of **forward declaration**: something that tells us that there will be a function with this signature defined somewhere in the program is without telling us exactly what it does. We can also have declarations for things other than functions in C++, as we shall see later on in the course. 

**With a function declaration:**
```cpp
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
```cpp
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
```cpp
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

## Header Files

Forward declarations for functions are helpful, but they can still clutter up our code if we are making use of large numbers of functions. We would also need to rewrite these forward declarations for _every_ source file that needs to use them! Instead, we put these declarations in **header files**, which usually end in `.h` or `.hpp`. We use `#include` to add header files to a `.cpp` file: this allows the file to get all the declaration from the header file. The definitions are not kept in the header file, they are in a separate `.cpp` file so that they can be compiled separately.  

In this case the files look as follows:

**function.h**:
```cpp
int f(int a, int b);  // function declaration
```

**function.cpp**:
```cpp
int f(int a, int b)
{
    return (a + 2) * (b - 3);
}
```

**main.cpp**:
```cpp
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

