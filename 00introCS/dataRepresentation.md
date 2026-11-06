---
title: Data Representation
---

# Background: Basic Data Representation

In order to write effective scientific programs, you will need to be able to translate real world and mathematical concepts into some kind of data representation. This is arguably the most fundamental process to understand in computing, since nothing can be worked with that cannot be represented. You may find that standard representations for many common objects (like vectors, complex numbers etc.) already exist, but it is absolutely vital that you understand how to construct these yourself. In this section, we will look at some examples of how to translate concepts into data representations in a language agnostic way.

## What is a data representation? 

In order to represent something a computer we need to have a properly formalised idea of exactly what information need to be stored and how. Computers themselves have only sequences of zeros and ones to work with, and the computer itself doesn't know what it is you're trying to represent with it and can't make any decisions on its own about the correct thing to do with them. 

Later in this course we will look at the object-oriented approach of combining data and functionality together, but for now we will restrict ourselves to the more fundamental topic of data alone. 

## Defining structures 

Programming languages usually provide us with a number of basic types of data that we can utilise, including:

- integers
- real numbers (approximated as "floating point")
- characters and strings
- arrays / lists

Almost all languages, including C++ and Python, will give us ways of bundling data together into a larger structure that a collection of named _fields_ of various types. In C++ and Python these are called _classes_[^struct-class]. Deciding what fields are needed for a class is a a fundamental skill in translating your ideas into computable data.

In C++, we must choose explicitly both the name and the _type_ of the data being held: is it an integer, or string, a floating point number? This is because C++ is _statically typed_ (variables must have a well known type that cannot change over the course of the program) whereas a language like Python is _dynamically typed_ (variables have the type of whatever data is assigned to them, and their type can change when their data is updated). 

[^struct-class]: Some of you may be familiar with _structs_ from C; the `struct` keyword from C is also valid in C++ and is also used to define a class. When declaring a data type with `struct` the members are public by default, whereas when using `class` the members are private by default; this is the only distinction made in C++ between these two keywords. 

## Example 1a: a circle

Suppose our model requires us to represent a circle in a 2D space. This could be because we need to draw it for a graphical output, or we need to keep track of everything that happens within a given radius of some object, or perhaps we need to do collision detections for a physical simulation. What information do we need to store in order to represent a circle on a computer? 

In order to answer this question, we should start by thinking about the least information necessary to fully define a circle: it will need a position $(x,y)$ and a radius, $r$. Everything else about a circle can be derived from this information: the circumference, the area, whether something is inside or outside of the circle, and so on. In a purely mathematical sense, $x$, $y$, and $r$ are all _real numbers_; on a computer our best way to represent real numbers is _floating point_ numbers, i.e. `float` or `double`. A class to represent a circle could look like this in C++:

```cpp
class Circle
{
    public:
    double centre_x;
    double centre_y;
    double radius;
};
```

or like this in Python:

```python
class Circle:
    centre_x: float
    centre_y: float
    radius: float
```

Note that in Python we have included the optional type annotations for clarity, and that `float` in Python has double precision and is therefore equivalent to `double` in C++. 

## Example 1b: adding colour to a circle

Consider now the graphical application of our circle class: we don't just need to know where it is and how big it is any more, but also what colour to render it. A colour on the computer is usually represented as a triple $(r, g, b)$ of _red_, _green_, and _blue_ intensities from 0 to 255. We can write a simple class to represent a colour then in the following way:

```cpp
class Colour
{
    public:
    int r;
    int g;
    int b;
}
```

If we've defined a `Colour` class, we can add it to our `Circle` class:

```cpp
class Circle
{
    public:
    double centre_x;
    double centre_y;
    double radius;
    Colour colour;
}
```

This kind of nested data structure is very helpful in building up complex structures, and more more readable than giving the `Circle` class its own separate red, green, and blue member variables. 

The Python equivalent would be:

```python
class Colour:
    r: int
    g: int
    b: int

class Circle:
    centre_x: float
    centre_y: float
    radius: float
    colour: Colour
```