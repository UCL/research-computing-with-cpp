---
title: Library Basics
---

## Library Basics

### Reviewing the build process

When building an application there are three important steps that the compiler must execute:

- **preprocessing**: follow the directives (lines started by `#` such as `#include` or `#define`) on the files to combine the units into what's passed to the compiler;
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

Directly using the compiler without a build tool (e.g., [CMake][lesson-cmake]) will eventually become too difficult and cause a mess, so we use CMake for larger projects!


You can find out more about preprocessor directives in [The C Preprocessor][CppAdv6] chapter on the [C++: Advanced Topics][CppAdv] course.

### Including libraries

You need your compiler to find:

* Headers: `#include`
* Libraries:
    * Dynamic: `.dylib` (mac), `.so` (linux), `.lib` / `.dll` (windows)
    * Static: `.a` (\*nix), `.lib` (windows)

We will see more about [differences between dynamic and static libraries][lesson-DynVsSt] later, but in brief:
- A **static** library is compiled with your program and included in your executable code. If you want to change the library behaviour by altering the library or upgrading to a new version, you need to re-compile your exectuable.
- A **dynamically linked** (a.k.a. **shared**) library is compiled separately into a a special kind of library object. You tell the compiler where this object is when you are _linking_, but the library itself is not part of your executable. You can change the library behaviour independently by updating and recompiling the library object, but if the library object is removed or you move your executable to a new system where it can't find the library object, then your executable will no longer work because it doesn't have all the code that it needs to run. Your executable will also not work if you change the dynamic library so that it no longer provides the necessary interface e.g. if you change a function signature in the library that the executable depends on. 
- Dynamically linked libraries are also called "shared libraries" because the same library object can be used by multiple executables, so you only need one copy of the compiled library code. On the other hand, if a static library is used by multiple executables then there will be copies of that library code in each of the executables. This can use more space, but makes it easier to keep executables independent and means that executables can be more easily maintined with different _versions_ of the same library without conflicts. 

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
  - Depending on your compiler, there are some standard locations where it will look for [includes](https://gcc.gnu.org/onlinedocs/gcc-4.9.4/cpp/Search-Path.html) or libraries (using `/usr/lib...` or `/usr/local/lib...`), but other locations may need to be provided using `-I` or `-L` flags to let it know where to look.
* Is it a [Header only][header-only-wiki]?
* What about bugs? How do I upgrade? Do I need to build it myself?

### Compilation Issues

Also, there are some issues related with the compilation:

* Which library version?
* Is there a requisite compiler version / C++ standard?
* Debug or Release?
* Static or Dynamic?
* 32 bit / 64 bit?
  - Most 64-bit machines can also run 32-bit code, but not the other way around! You should take advantage of 64-bit compilation where you can though. 
* Platform specific flags?
* Pre-installed, or do you compile it?

### Wrapping: a technique for avoiding library pain

Some libraries it's obvious that we've made the correct choice, perhaps we've used a library before or someone we trust has recommended it. Other libraries we can be a little more nervous about, perhaps we're not sure it does what we need, or we're worried that in the future we'll need to swap it out with a different library or write our own.

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
[Cherno]: https://www.youtube.com/channel/UCQ-W1KE9EYfdxhL6S4twUNw
[lesson-DynVsSt]: ./sec02LinkingLibraries.html
[header-only-wiki]: https://en.wikipedia.org/wiki/Header-only
[learncpp-static-dynamic]: http://www.learncpp.com/cpp-tutorial/a1-static-and-dynamic-libraries/
