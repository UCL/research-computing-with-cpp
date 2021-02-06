---
title: Library Basics
---

## Library Basics

### Reviewing the build process

When building an application there are three important steps:
- **preprocessing**: follow the directives (lines started by `#`) on the files to combine the units into what's passed to the compiler;
- **compilation**: translates the program into machine language code - object files;  and
- **linking**: merges the various object files and links all the referred libraries as needed to create the executable.

Though normally these steps are invoked by a single command, you can run them one at a time.
- preprocessor:
  ```bash
  g++ -E -o <output> <input.cpp>
  ```
- compilation:
  ```bash
  g++ -c -o <output.o> <input.cpp>
  ```
- linking:
  ```bash
  g++ -o <executable> <output.o>
  ```
Directly using the compiler without a build tool (e.g., [CMake][lesson-cmake]) will eventually become too difficult and cause a mess.


Find more details of these steps on the following material:
- [The C Preprocessor][CppAdv6] chapter on the [C++: Advanced Topics][CppAdv] course.
- How the C++ [Compiler][CppChernoCompiler] and [Linker][CppChernoLinker] works videos by [The Cherno][Cherno].

### Including libraries

You need your compiler to find:

* Headers: `#include`
* Libraries:
    * Dynamic: `.dylib` (mac), `.so` (linux), `.lib` / `.dll` (windows)
    * Static: `.a` (*nix), `.lib` (windows)

We will see the [differences between dynamic and static libraries][lesson-DynVsSt] in the next page.
Let's see first how we include the libraries to our programme.

### In practice

Normally, when including with `<>` the preprocessor looks for headers in the include path list.
You can specify the include folder(s) by using the `-I` argument as needed.
Similarly, the `-L` argument is used to give the path in which the linker should search for libraries to link,
and the `-l` flag gives the name of the library to be linked. Note that the libraries files always start with `lib`
but we don't add such prefix when referring to it.

For example:

```bash
# compilation
g++ -c -I /users/me/myproject/include main.cpp
# linking
g++ -o main main.o -L /users/me/myproject/lib -l mylib
```

#### Check Native Build Platform

* Look inside
    * Makefile
    * Visual Studio options
* Basically constructing `-I`, `-L`, `-l` switches to pass to command line compiler.


#### Windows Compiler Switches

* Visual Studio (check version)
* Project Properties
    * C/C++ -> Additional Include Directories.
    * Configuration Properties -> Linker -> Additional Library Directories
    * Linker -> Input -> Additional Dependencies.
* Check compile line - its equivalent to Linux/Mac, -I, -L, -l


### Location Issues

When you use a library:

* Where is it?
* Header only?
* System version or your version?
* What about bugs? How do I upgrade? Do I need to build it myself?


### Compilation Issues

When you use a library:

* Which library version?
* Which compiler version?
* Debug or Release?
* [Static or Dynamic?](http://www.learncpp.com/cpp-tutorial/a1-static-and-dynamic-libraries/)
* 32 bit / 64 bit?
* Platform specific flags?
* Pre-installed, or did you compile it?


### A Few Good Libraries

Due to all those issues shown above, again, the main advice for libraries:

* As few as possible
* As good as possible

[CppAdv]: https://www.linkedin.com/learning/c-plus-plus-advanced-topics/
[CppAdv6]: https://www.linkedin.com/learning/c-plus-plus-advanced-topics/about-the-preprocessor
[lesson-cmake]: ../01research/sec04CMakeBasics.html
[CppChernoCompiler]: https://www.youtube.com/watch?v=3tIqpEmWMLI
[CppChernoLinker]: https://www.youtube.com/watch?v=H4s55GgAg0I
[Cherno]: https://www.youtube.com/channel/UCQ-W1KE9EYfdxhL6S4twUNw
[lesson-DynVsSt]: ./sec02LinkingLibraries.html
