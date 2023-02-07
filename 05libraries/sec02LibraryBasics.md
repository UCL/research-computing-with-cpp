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


Find more details of these steps on the following resources:

- [The C Preprocessor][CppAdv6] chapter on the [C++: Advanced Topics][CppAdv] course.
- How the C++ [Compiler][CppChernoCompiler] and [Linker][CppChernoLinker] works videos by [The Cherno][Cherno].

### Including libraries

You need your compiler to find:

* Headers: `#include`
* Libraries:
    * Dynamic: `.dylib` (mac), `.so` (linux), `.lib` / `.dll` (windows)
    * Static: `.a` (\*nix), `.lib` (windows)

We will see the [differences between dynamic and static libraries][lesson-DynVsSt] later. Let's see first how we include the libraries in our code.

### In practice

Normally, when including with `<>` the preprocessor looks for headers in the *include path list*. You can specify the include folder(s) by using the `-I` argument as needed. Similarly, the `-L` argument is used to give the path in which the linker should search for libraries to link, and the `-l` flag gives the name of the library to be linked. Note that the libraries files always start with `lib` but we don't add such prefix when referring to it.

For example:

```bash
# compilation
g++ -c -I /users/me/myproject/include main.cpp
# linking
g++ -o main main.o -L /users/me/myproject/lib -l mylib
```

### Location Issues

When you use a library, keep in mind the following questions:

* Where is it? (do we need to set `-I` and `-L` when compiling?)
* Is it a [Header only][header-only-wiki]?
* System's version or your version?
* What about bugs? How do I upgrade? Do I need to build it myself?

### Compilation Issues

Also, there are some issues related with the compilation:

* Which library version?
* Which compiler version?
* Debug or Release?
* [Static or Dynamic][learncpp-static-dynamic]?
* 32 bit / 64 bit?
* Platform specific flags?
* Pre-installed, or did you compile it?

### Wrapping: a technique for avoiding library pain

Some libraries it's obvious that we've made the correct choice, perhaps we've used a library before or someone we trust has recommended it. Other libraries we can be a little more nervous about, perhaps we're not sure it does what we need, or we're worried that in the future we'll need to swap it out with a different library (or worse, write our own!).

If we think we might need to swap the library out at some future stage, we can *wrap* the library, creating an interface between the library and our own code. This minimises the number of places we must change our code if we ever need to change the library, and we can augment the library to suit our needs. For example, if we know we want to load JSON files, but we're not sure which library to use, we could choose one library, e.g. `json_library`, and write a wrapper around it:

```cpp
#import <json_library>

string readJson(const string& filename) {
  return json_library::read(...);
}
```

Here, we've just written a example `readJson` function that reads some JSON from a file, which we use in our own code instead of calling the library directly. If we ever need to change the library, we can simply change the code inside this function instead of every place we read a JSON file. Wrappers can be simple functions like this, or whole classes that do more than just provide a new interface.

Of course, wrappers are not free; they're more code to write, test, document and maintain, but they can help protect you from future changes to libraries.
    

[CppAdv]: https://www.linkedin.com/learning/c-plus-plus-advanced-topics/
[CppAdv6]: https://www.linkedin.com/learning/c-plus-plus-advanced-topics/about-the-preprocessor
[lesson-cmake]: ../01projects/sec04CMakeHelloWorld.html
[CppChernoCompiler]: https://www.youtube.com/watch?v=3tIqpEmWMLI
[CppChernoLinker]: https://www.youtube.com/watch?v=H4s55GgAg0I
[Cherno]: https://www.youtube.com/channel/UCQ-W1KE9EYfdxhL6S4twUNw
[lesson-DynVsSt]: ./sec02LinkingLibraries.html
[header-only-wiki]: https://en.wikipedia.org/wiki/Header-only
[learncpp-static-dynamic]: http://www.learncpp.com/cpp-tutorial/a1-static-and-dynamic-libraries/
