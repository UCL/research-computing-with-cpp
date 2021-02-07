---
title: Using Libraries
---

## Using libraries

Now that we understand what libraries are and how we can access to them, let's
see some practicalities about using them.

### Package Managers (*nix)

Installing libraries using a Package Manager (Linux/Mac) has some advantages:
* they are pre-compiled,
* provide a Stable choice, and
* inter-dependencies work.

For Linux you can use:
* `sudo apt install` for debian based systems,
* `sudo dnf install` for rpm systems,
* or whichever [package manager][linux-pm-wiki] your system uses.

macOS has also many options, though they need to be installed, for example:
* [homebrew][homebrew]: `brew install`
* [MacPorts][macports]: `sudo port install`


### Windows

On Windows, the libraries typically are:
* on randomly installed locations, or
* in system folders, or
* in the developer's folders, or
* in the build folder.

The absence of a "standard" approach makes that our machine could become
full of mixed libraries. Be careful and try to keep your machine clean.
We suggest you invest some time exploring Windows package managers such as
[Chocolatey][chocolatey] and [winget][winget]


### Package Managers - summary

In summary, if you can use standard versions of 3rd party libraries, then
Package Managers are a good way to go. You just need to specify what versions
your program depends on so your collaborator can install it too.


#### Problems

However, you may encounter some problems. What happens if you
find a bug in a library?

Is there an easy way to update? Does that update produces a
know on effect? (e.g., cascading updates and produces inconsistent
development environments)

### Build Your Own

An alternative solution to use "stable" libraries provided by your system is
to build your own.

This can be done on two ways:
* Using External / Individual builds
    * Build each dependencies externally and
    * Point your software at those packages
* SuperBuild / Meta-Build
    * Write code to download and build all dependencies
    * Storing the correct version numbers in the code

C++ package managers makes it easy to generate SuperBuilds.

Let's see examples for each case:


#### External / Individual Build

For example, let's say that we've got builds for a particular version of the
ITK and VTK libraries on a particular directory of our computer `C:\build` and
this is how our tree looks:

```
C:\build\ITK-v1
C:\build\ITK-v1-build
C:\build\ITK-v1-install
C:\build\VTK-v2
C:\build\VTK-v2-build
C:\build\VTK-v2-install
C:\build\MyProject
C:\build\MyProject-build
```

Our project `MyProject-build` needs to know the location of ITK and VTK install
folder. We will see in the [next page][lesson-lib-example] how to do so.


#### Meta-Build / Super-Build

Alternatively, we could set our project to download and build all the dependencies
within our project, so our tree would look now like:

```
C:\build\MyProject
C:\build\MyProject-SuperBuild\ITK\src
C:\build\MyProject-SuperBuild\ITK\build
C:\build\MyProject-SuperBuild\ITK\install
C:\build\MyProject-SuperBuild\VTK\src
C:\build\MyProject-SuperBuild\VTK\build
C:\build\MyProject-SuperBuild\VTK\install
C:\build\MyProject-SuperBuild\MyProject-build
```

`MyProject-build` knows now the location of ITK and VTK as were compiled by itself.


#### Pro's / Con's

Each methods has its own advantages and disadvantages, and you should choose the
most appropriate for your use case.

External Build:
* Pro's - build each dependency once
* Con's - collaborators will do this inconsistently
* Con's - how to manage multiple versions of all dependencies

Meta Build:
* Pro's - all documented, all self-contained, easier to share
* Con's - Slow build? Not a problem if you only run `make` in sub-folder
  `MyProject-build`. (You can also use [`ccache`][ccache] on clean builds to
  speed compilation up).

[linux-pm-wiki]: https://en.wikipedia.org/wiki/Package_manager
[homebrew]: https://brew.sh/
[macports]: https://www.macports.org/
[chocolatey]: http://chocolatey.org
[winget]: https://docs.microsoft.com/en-us/windows/package-manager/
[lesson-lib-example]: ./sec04Examples.html
[ccache]: https://ccache.dev/
