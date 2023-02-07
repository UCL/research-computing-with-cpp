---
title: Installing Libraries
---

## Installing libraries

Now that we understand what libraries are and how we use them in our code, let's discuss how to install libraries so we can use them. The main ways we can install libraries are:

- with a C++ package manager (e.g. conan)
- with the system package manager (e.g. `apt`, `yum`, `pacman`)
- including and building the library with our code



### Package managers (C++)

Package managers like [conan][conan] can make installing C++ packages as simple as installing Python packages with `pip`. Take a look at the excellent [CMake tutorial with Conan](https://docs.conan.io/en/2.0/tutorial/consuming_packages/build_simple_cmake_project.html).

### Package managers (\*nix)

Installing libraries using a package manager (Linux/Mac) has some advantages:

* they are pre-compiled,
* provide a stable choice, and
* inter-dependencies work.

However, you have relatively little control over the version of the library you install and you add extra steps to a user's experience if they wish to use your code.

For Linux you can use:

* `sudo apt install` for debian based systems,
* `sudo dnf install` for rpm systems,
* or whichever [package manager][linux-pm-wiki] your system uses.

macOS has also many options, though they need to be installed, for example:

* [homebrew][homebrew]: `brew install`
* [MacPorts][macports]: `sudo port install`

### Package managers (Windows)

On Windows, the libraries typically are:

* on randomly installed locations, or
* in system folders, or
* in the developer's folders, or
* in the build folder.

The absence of a "standard" approach makes that our machine could become
full of mixed libraries. Be careful and try to keep your machine clean.
We suggest you invest some time exploring Windows package managers such as
[Chocolatey][chocolatey] and [winget][winget]

### Using system libraries from CMake

Many libraries now support CMake's modules. These make it extremely easy to find and use libraries within CMake. Let's illustrate that with the graphics library SFML. When installed via the system package manager, it exposes a `FindSFML.cmake` file that CMake can use to simply import the library in our own project. The way we do that in our `CMakeLists.txt` file is to add the line:

```
find_package(SFML 2 REQUIRED network audio graphics window system)
```

In the particular case of SFML, we need include files and library files, which we tell CMake to add to our target executable `hello` with the lines:

```
target_include_directories(hello PUBLIC ${SFML_INCLUDE_DIR})
target_link_libraries(hello PUBLIC ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
```

And that's it! Another short example is OpenMP (which we will be using later in the course):

```
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hello PUBLIC OpenMP::OpenMP_CXX)
endif()
```

As long as the library provides CMake integration, it's usually this simple to link it to your project.

For further reading on importing libraries with CMake, check out some [good practices in using cmake modules](https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1#modules).

### Using FetchContent within CMake

Although I recommend using a package manager like Conan to best manage dependencies, CMake also provides a way to download and import missing dependencies using its `FetchContent` feature, which you can find out more about in [this tutorial on FetchContent](https://coderefinery.github.io/cmake-workshop/fetch-content/).

[linux-pm-wiki]: https://en.wikipedia.org/wiki/Package_manager
[homebrew]: https://brew.sh/
[macports]: https://www.macports.org/
[chocolatey]: http://chocolatey.org
[winget]: https://docs.microsoft.com/en-us/windows/package-manager/
[lesson-lib-example]: ./sec04Examples.html
[ccache]: https://ccache.dev/
