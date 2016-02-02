---
title: Boost
---

{% idio cpp %}

## Using Boost

### Introduction

* [Boost][BoostHome] is "...one of the most highly regarded and expertly designed C++ library projects in the world."
* A [large (121+)][BoostDoc] collection of C++ libraries
* Aim to establish standards, contribute to C++11, C++17 etc.
* Hard to use C++ without bumping into Boost at some point
* It's heavily templated
* Many libraries header only, some require compiling.


### Libraries included

* Log, FileSystem, Asio, Serialization, Pool (memory) ...
* Regexp, String Algo, DateTime,  ...
* Math, Odeint, Graph, Polygon, Rational, ...
* Each has good documentation, and tutorial, and unit tests, and is widely compiled.


### Getting started

* Default build system: ```bjam```
* Also CMake version of boost project, possible deprecated.
* Once installed, many header only libraries, so similar to Eigen.


### Installing pre-compiled

* Linux:
    * ```sudo apt-get install boost```
    * ```sudo apt-get install libboost1.53-dev```
* Mac
    * Homebrew (brew) or Macports (port)
* Windows
    * Precompiled binaries? Probably you need to build from source.


### Compiling from source

* [Follow build instructions here][BoostBuild]
* Or use bigger project with it as part of build system
    * NifTK, MITK, Slicer, Gimias (medical imaging)


### C++ Principles

(i.e. why introduce Boost on this course)

* Boost uses
    * Templates
    * Widespread use of:
        * Generic Programming
        * Template Meta-Programming
    * Functors


### C Function Pointers - 1

* Useful if using [Numerical Recipes in C][NumericalRecipesC]
* See [Wikipedia][WikipediaFunctionPointers] article and tutorials online

This:

{% code FunctionPointer/FunctionPointer.cc %}

Produces:

{% code FunctionPointer/FunctionPointer.out %}


### C Function Pointers - 2

* Function pointers can be passed to functions

This:
{% code FunctionPointer/PassToFunction.cc %}

Produces:
{% code FunctionPointer/PassToFunction.out %}


### C Function Pointers - 3

* Function pointers
    * often used for callbacks, cost functions in optimisation etc.
    * called by name, or dereference pointer
    * are generally stateless


### C++ Function Objects - 1

* We can define an object to represent a function
    * Called [Function Object][WikipediaFunctionObject] or Functor

This:

{% code FunctionObject/FunctionObject.cc %}

Produces:
{% code FunctionObject/FunctionObject.out %}


### C++ Function Objects - 2

* But function objects can
    * Have state
    * Have member variables
    * Be complex objects, created by any means
    * e.g. Cost function, similarity between two images


### CMake for Boost

* If installed correctly, should be something like:

``` cpp
set(Boost_ADDITIONAL_VERSIONS 1.53.0 1.54.0 1.53 1.54)
find_package(Boost 1.53.0)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()
```


### Boost Example

* With 121+ libraries, can't give tutorial on each one!
* Pick one small numerical example
* Illustrate the use of functors in ODE integration


### Using Boost odeint

* Its a numerical example, as we are doing scientific computing!
* As with many libraries, just include right header

``` 
#include <boost/numeric/odeint.hpp> // Include ODE solver library
                                    // just to check our build system found it
```
* See [this tutorial][BoostTutorial]


### Boost odeint - 1

Given these global definitions:

{% fragment global, Boost/BoostHarmonicOscillator.cc %}


### Boost odeint - 2

First define a functor for the function to integrate:

{% fragment harm_osc, Boost/BoostHarmonicOscillator.cc %}


### Boost odeint - 3

Define an observer to collect graph-points:

{% fragment observer, Boost/BoostHarmonicOscillator.cc %}


### Boost odeint - 4

The run it:

{% fragment main, Boost/BoostHarmonicOscillator.cc %}


### Boost odeint - 4

Produces:

{% code Boost/BoostHarmonicOscillator.out %}


### Why Boost for Numerics

* Broader question is
    * Why someone else's library? Boost or some other.
* Advanced use of Template Meta Programming, Traits
    * Performance optimisations
    * Alternative implementations
        * CUDA via Thrust
        * MPI
        * etc
* You just focus on your bit

{% endidio %}

[BoostHome]: http://www.boost.org/
[BoostDoc]: http://www.boost.org/doc/libs/1_57_0/
[BoostBuild]: http://www.boost.org/doc/libs/1_57_0/libs/regex/doc/html/boost_regex/install.html
[BoostTutorial]: http://www.boost.org/doc/libs/1_57_0/libs/numeric/odeint/doc/html/index.html
[NumericalRecipesC]: http://www.nr.com/
[WikipediaFunctionPointers]: http://en.wikipedia.org/wiki/Function_pointer
[WikipediaFunctionObject]: http://en.wikipedia.org/wiki/Function_object
