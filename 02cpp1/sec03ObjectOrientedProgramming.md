---
title: Object Oriented Programming
---

Estimated Reading Time: 60 minutes

# Custom Types and Object Oriented Programming (OOP) in C++

As a programming language, C++ supports multiple styles of programming, but it is generally known for _object oriented programming_, often abbreviated as _OOP_. This is handled in C++, as in many languages, through the use of classes: special datastructures which have both member data (variables that each object of that class contains and which are usually different for each object) and member functions, which are functions which can be called through an object and which have access to both the arguments passed to it _and_ the member variables of that object. 

We have already been making extensive use of classes when working with C++. Indeed, it is difficult not to! The addition of classes was the main paradigm shift between C, a procedural programming language with no native support for OOP, and C++. 

## Classes

Classes can be used to define our own data-structures, which have their own type. We can then declare objects of this type in our program. Apart from a handful of built in types (like `int`, `double`, and `bool`), variables that we declare in C++ are instances of a class. A number of objects that we've used so far are classes defined in the standard library, like `vector` and `string`. 

Classes achieve two goals in representing concepts in programming: 

- _Abstraction_
    - Represents the essential elements of a _kind_ of object, as distinct from other kinds of objects. What are the defining properties of a type of object? 
    - Class defines the blueprint for every object of that kind: what information it contains and what it should be able to do. 
    - Objects are individual instances of a class. 
    - _“An abstraction denotes the essential characteristics of an object that distinguish it from all other kinds of objects and thus provide crisply defined conceptual boundaries, relative to the perspective of the viewer.”_ - Grady Booch
- _Encapsulation_
    - Methods and data that belong together and kept together. 
    - Provide public interface to class: how other things should be able to interact with it.
    - Protects and hides data to which other things should not have access. 

## Access Specifiers in Classes

When writing a class we can declare a member function or variable using one of three access specifiers:

- `private`: access is private by default. The variable or function is available only within the body of this class. 
- `protected`: The variable or function can be accessed within the body of this class, or within the body of any class which inherits from this class. 
- `public`: The variable or function can accessed inside and outside of the definition of the class, by anything which can access the object. 

The access specifiers, `private`, `protected`, and `static`, are keywords which are used within class definitions followed by a colon (`:`) to specify access for all following members until the end of the class or another access specifier is reached. For example:

```cpp
class myClass
{
    public:
    int x;
    double y;

    private:
    std::string name;

    protected: 
    double z;
};
```

- `x` and `y` are both public
- `name` is private
- `z` is protected

If you are writing classes in C++, especially classes that will be used by other people, it's a good idea to only give people access to as much as they need and no more than that. In general:

- Make functions and variables `private` if you can.
- You can control access to variables in a finer grained way through `get` and `set` methods than by making them public. For example you may want variables that can be inspected (write a `get` function) but not changed (no `set` function) or vice versa. 
- Constructors and destructors should generally be `public`. 


## Static Members 

Static member variables or functions are special members of a class. They belong to the class as a whole, and do not have individual values or implementations for each instance. This can be useful when keeping track of properties that are changeable and may affect the class as a whole, or for keeping track of information about a class. For example, one can use a static variable to count the number of instances of a class which exist using the following:

```cpp
class countedClass
{
    public:

    countedClass()
    {
        count += 1;
    }

    ~countedClass()
    {
        count -= 1;
    }

    static int count;
};

int countedClass::count = 0;

int main()
{
    auto c1 = countedClass();
    cout << countedClass::count << endl;

    auto c2 = countedClass();
    cout << c2.count << endl;

    return 0;
}

```
- The count is incremented in the constructor (`countedClass()`), and so increased every time an instance of this type is created. 
- The count is decremented in the destructor (`~countedClass()`), and so decreased every time an instance of this type is destroyed. 
- `count` is a static variable, so belongs to the class as a whole. There is one variable `count` for the whole class, regardless of how many instances there are. The class still accesses it as a normal member variable. 
- `count` also needs to be declared outside of the class definition. (This is where you should initialise the value.) 
- A static variable can be accessed in two different ways: through the object (`c1.count`), or through the class namespace (`countedClass::count`) without reference ot any object. Public static variables for a class can therefore be accessed by anything which has access to the class definition, regardless of whethere there are any objects of that class. 

## Improving this class with Access Specifiers

- A variable like `count` shouldn't be able to be changed outside of the class, as that could interfere with our counting! But we do want to be able to access the _value_ of the count, so we can tell how many there are. 
- We should make `count` _private_ and make a function to retrieve the value _public_
- Such functions are often called "getters", because they are frequently named `get...()` for some variable

```cpp
class countedClass
{
    public:

    countedClass()
    {
        count += 1;
    }

    ~countedClass()
    {
        count -= 1;
    }

    static int getCount()
    {
        return count;
    }

    private:
    static int count;
};

int countedClass::count = 0;

int main()
{
    auto c1 = countedClass();
    cout << countedClass::getCount() << endl;

    auto c2 = countedClass();
    cout << c2.getCount() << endl;

    return 0;
}
```

- `getCount()` is `public` and `static` and so can be accessed just like we accessed `count` before (through an object or through the class definition).
- `getCount()` returns an integer _by value_, so it returns a copy of `count`. We can't modify `count` through this function or the value we get back from it. 
- `count` is now private, so if we try to access this directly from outside the class the compiler will raise an error. 

## Class Invariants and Using OOP for Data Integrity

An extremely useful aspect of defining a new type via a class is the ability to provide guarantees that any object of that type satisfies certain properties; such properties are often referred to as class _invariants_. These properties allow programmers to write programs that are more efficient and correct with less overhead for error checking. 

Let's explore this with some examples. 

### Ensuring Objects Are Self-Consistent

Let's suppose we have a physical simulation, which involves a ball suspended in some fluid. A ball will probably have the following fields:
```cpp
class Ball
{
    std::array<double, 3> position;
    double radius;
    double mass;
};
```

These fields define the sphere well, but physically the behaviour of the sphere in the fluid will depend on its _density_. So perhaps we want to write a member function `double density(double radius, double mass)` which calculates the density of the sphere. But this would mean we need to call the density function and re-calculate it when we want to use it, which isn't ideal. So instead, we can add density to our list of fields, 
```cpp
class Ball
{
    public:
    std::array<double, 3> position;
    double radius;
    double mass;
    double density;
};
```

and then we can call the density directly without another calculation. The problem that we now have is that in order for our data to be self-consistent, **a relationship between the radius, mass, and density must be satisfied**. This kind of relationship is called an **invariant**: a property that must be maintained by all instances of a class. Invariants are very important for writing safe programs, and for being able to reason about the behaviour of programs. 

We could approach this problem by calculating the density in the constructor, and making the radius, mass, and density **private**. This means that external code can't change any of these values, and therefore they can't become inconsistent with one another. But we still need to be able to _read_ these variables for our physics simulation, so we'll need to write **getter** functions for them:
```cpp
class Ball
{
    public:
    Ball(std::array<double, 3> p, double r, double m): position(p), radius(r), mass(m)
    {
        setDensity();
    }
    
    std::array<double, 3> position;
    double getRadius(){return radius;}
    double getMass(){return mass;}
    double getDensity(){return density;}

    private:
    void setDensity()
    {
        density =  3 * mass / (4 * M_PI * pow(radius, 3));
    }
    double radius;
    double mass;
    double density;
};
```

Now we can even make our code **more flexible without sacrificing safety**. Let's say the ball can change _mass_ or _radius_. We can't just make these variables public and change them independently, because then the _density_ will no longer be consistent with the new mass / radius. We need to add **setter** functions which **maintain the integrity of the object**:
```cpp
class Ball
{
    public:
    Ball(std::array<double, 3> p, double r, double m): position(p), radius(r), mass(m)
    {
        setDensity();
    }

    std::array<double, 3> position;
    double getRadius(){return radius;}
    double getMass(){return mass;}
    double getDensity(){return density;}

    double setRadius(double r)
    {
        radius = r;
        setDensity();
    }

    double setMass(double m)
    {
        mass = m;
        setDensity();
    }

    private:
    void setDensity()
    {
        density =  3 * mass / (4 * M_PI * pow(radius, 3));
    }

    double radius;
    double mass;
    double density;
};
```
We now have a ball class that can be instantiated with any mass and radius, and can have its mass or radius changed, but **always satisfies the property that the density field is correct for the given radius and mass of the object**. Being able to guarantee properties of objects of a given type makes the type system far more powerful and gives users the opportunity to use objects in more efficient ways without having to check for conditions that are already guaranteed by the object's design. 

### Maintaining Desirable Properties

Consider another example where we have a catalogue for a library. To keep things simple, we'll say that we just store the title of each book. Very simply, we could define this as a vector:
```cpp
vector<string> catalogue;
```
and every time we want to add a new title we can simply stick it on the end of the list:
```cpp
catalogue.push_back("Of Mice and Men");
```
Adding books to our catalogue is certainly very simple! But what happens when we want to _look up_ a book, to see if it's in the catalogue? 

In an unordered list, the only thing we can do is go through the entire list one by one until we find it or reach the end of the list. The amount of time that we take searching will be proportional to the length of our catalogue, which isn't great performance. 

This is particularly bad because we'd expect people to look up books far more often than we add new ones! How can we do better?

If our list were _sorted_, then we can search much more quickly using a [binary search](https://en.wikipedia.org/wiki/Binary_search_algorithm). A binary search on a sorted list starts by looking at the element in the middle of the list and checks if the item we're looking for should come before or after that. We then only need to search the half of the list that would contain the book we're looking for. We then apply the same thing again to narrow the list down by half again, and so on. At every step we half the size of the list and therefore the number of titles we have to check is proportional to _the logarithm of the size of the list_. This is much, much better performance, especially if the size of the list is large. A binary search with 21 comparisons could search a list of over a million books! 

Of course, we don't want to sort our data before searching it every time (that would be even more wasteful than our linear search), and we want to know with certainty that our list is always sorted, otherwise our binary search could fail. Using an object is a solution: we can define a wrapper class which keeps the list private, and provides an insertion method which guarantees that new entries are inserted into their proper place. Then **we can take advantage of speedier lookup because we know that our catalogue is always in sorted order**. In this case our _invariant_ is the property of being sorted, or put more explicitly $i < j \implies x_i \le x_j$. (Incidentally, this would normally be done with a _balanced binary search tree_, an example of which is the C++ `map` type.)

### Reasoning About Class Invariants

From these examples we can see an important pattern arise: an object will maintain the desired property if it is constructed in a state which has that property, and if all permissible operations on the object maintain that property. This is a form of _inductive reasoning_, where the initial construction of the object serves as a base case, and all other possible states of the object are found by the operations on that object (calling member functions or manipulating public data). To design a class where any object of that class maintain a property $P$ then you should:

- Write you constructor so that $P$ is guaranteed for any constructed object. Be wary of uninitialised variable within your class. 
- Make any variables `private` if a modification of that variable can alter the property $P$. For example, to maintain a list as being sorted we made the underlying `vector` private because any modification of the data in the array could violate the sorting property. To protect our `Ball` class we made the `mass`, `radius`, and `density` private since modifying any one of these could violate the physical relationship between these parameters.
- Ensure that any functions that modify the state of the class do not violate the property. In the case of our sorted list, this means that the insertion must update the list in a way that it remains sorted. Be sure to check any setters, as with the `Ball` class: modifying any one of the properties of the ball has consequences for the others. It's a good idea to mark any functions that should not modify the state as `const` so that the compiler can spot any potential risks. 

## Aside: Organising Class Code in Headers and Source Files

As we saw last week, C++ code benefits from a separation of function declarations (in header files) and implementations (in source files) when these functions need to be included in other files. A similar principle applies to classes. 

In the header file, we should declare the class as well as:
1. What all of its member variables are
2. Function declarations for all of its member functions 
3. Can also include full definitions for trivial functions such as getter/setter functions

For example:
**In `ball.h`:**
```cpp
class Ball
{
    public:
    Ball(std::array<double, 3> p, double r, double m);
    
    std::array<double, 3> position;
    double getRadius(){return radius;}
    double getMass(){return mass;}
    double getDensity(){return density;}

    private:
    void setDensity();

    double radius;
    double mass;
    double density;
};
```
**In `ball.cpp`:**
```cpp
// constructor definition
// Ball:: tells us that the function Ball(...) is part of the Ball class
Ball::Ball(std::array<double, 3> p, double r, double m): position(p), radius(r), mass(m)
{
    setDensity();
}

// Again, Ball:: tells us that this function is part of the Ball class definition
// Because this is a member function, it has access to all the data members of this class.
void Ball::setDensity()
{
    density =  3 * mass / (4 * M_PI * pow(radius, 3));
}
```

We must include declarations for all member functions and variables in the class because any code which makes use of the class needs to know the full interface. It's also very important for C++ compilers to know what data a class needs to hold in order to know how much memory to reserve when constructing it. Because object files can be compiled separately, the information about data members must be in the header. 
