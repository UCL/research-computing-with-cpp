---
title: Object Oriented Programming
---

Estimated Reading Time: 60 minutes

# Custom Types and Object Oriented Programming (OOP) in C++

As a programming lanaguage, C++ supports multiple styles of programming, but it is generally known for _object oriented programming_, often abbreviated as _OOP_. This is handled in C++, as in many languages, through the use of classes: special datastructures which have both member data (variables that each object of that class contains and which are usually different for each object) and member functions, which are functions which can be called through an object and which have access to both the arguments passed to it _and_ the member variables of that object. 

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
- The count is incremented in the constuctor (`countedClass()`), and so increased every time an instance of this type is created. 
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

## Using Objects for Data Integrity

An extremely useful aspect of defining a new type via a class is the ability to provide guarantees that any object of that type satisfies certain properties. These properties allow programmers to write programs that are more efficient and correct with less overhead for error checking. 

Let's explore this with some examples. 

### Ensuring Objects Are Self-Consistent

Let's suppose we have a physical simulation, which involves a ball suspended in some fluid. A ball will probably have the following fields:
```cpp
class Ball
{
    std::array<double, 3> position;
    double radius;
    double mass;
}
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
}
```

and then we can call the density directly without another calculation. The problem that we now have is that in order for our data to be self-consistent, **a relationship between the radius, mass, and density must be satisfied**. 

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
}
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
}
```
We now have a ball class that can be instantiated with any mass and radius, and can have its mass or radius changed, but **always satisfies the property that the density field is correct for the given radius and mass of the object**. Being able to guarantee properties of objects of a given type makes the type system far more powerful and gives users the opportunity to use objects in more efficient ways without having to check for conditions that are already guaranteed by the object's design. 

### Maintaining Desireable Properties

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

Of course, we don't want to sort our data before searching it every time (that would be even more wasteful than our linear search), and we want to know with certainty that our list is always sorted, otherwise our binary search could fail. Using an object is a solution: we can define a wrapper class which keeps the list private, and provides an insertion method which guarantees that new entries are inserted into their proper place. Then **we can take advantage of speedier lookup because we know that our catalogue is always in sorted order**. (Incidentally, this would normally be done with a _balanced binary search tree_, an example of which is the C++ `map` type.)

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
}
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
Ball::setDensity()
{
    density =  3 * mass / (4 * M_PI * pow(radius, 3));
}
```

We must include declarations for all member functions and variables in the class because any code which makes use of the class needs to know the full interface. It's also very important for C++ compilers to know what data a class needs to hold in order to know how much memory to reserve when constructing it. Because object files can be compiled separately, the information about data members must be in the header. 

## Creating Sub-types with Inheritance

Inheritance is one of the most important concepts in object oriented design, which brings a great deal of flexibility to us as programmers. A class defines a type of object, and a class which inherits from it defines a sub-type of that type. For example, we might have a class which represents shapes, and sub-classes which represent squares, circles, and triangles. Each of these are shapes, and so should be able to be used in any context that simply requires a shape, but each will have slightly different data needed to define it and different implementations of functions to calculate its perimeter or area. 

If we have a class to represent shapes, then any function which takes an object of our shape class should be able to take a circle, a square, or a triangle. This ability to use different types in the same context is called **polymorphism** and is a key concept in many programming paradigms. In C++ one of the key ways we will achieve it is by using inheritance. 

## When Should Inheritance Be Used?

- Inheritance should be used only when you want to declare that one class is a sub-type of another class. Essentially **`B` may inherit from `A` only if `B` _is a kind of_ `A`.**
- A common example is that the classes `Circle` and `Square` may both derive from the class `Shape`. But neither `Circle` nor `Square` should inherit from one another! 
    - Consider for example a class `Country`, which may have both and area and a perimeter. Although it shares some properties with `Shape`, it should almost certainly **not** inherit from `Shape`, because a `Country` is not a kind of `Shape`, and we wouldn't expect a `Country` to be substitutable everywhere that a `Shape` is. This is an example of using the type system to our advantage: we shouldn't allow a `Country` to be passed into a `Shape` function, because we know it is the wrong kind of object even if it shares some (or even all) properties. We are using the type system to impart information that we understand about the objects we are creating and modelling, and discriminate between representations of different kinds of thing. 
- The **Liskov Substitution Principle** is one good guiding principle. 
    - If `B` is a sub-type of `A`, then replacing an object of type `A` with an object of type `B` should not break your program. 
    - In this case a `B` object can be considered a kind of `A` object, but not the other way around. 
    - Sub-types are more specific than base types. 
- Derived classes should generally _extend_ classes rather than restrict classes. Having a subclass that is simpler than the base class can cause problems if the object is substituted into a part of the program that expects a base class, as functionality expected for the base class may not be appropriately defined in the derived class. 
    - This is commonly referred to as the Square/Rectangle problem or the Circle/Ellipse Problem.
    - In this case the square (or circle) is a special case of a rectangle (or ellipse) which is more restricted an contains less information, because it only requires one length to be defined instead of two. 
    - Functionality that manipulates the height and width of the rectangle individually don't make sense for a square, because it should only have one. 

## Composition: When **not** to use Inheritance

- Don't use inheritance if you want a class to _have_ an instance of another class as a component.
    - It should be achieved by having a member variable of that type, or a pointer to an object of that type. 
        - For example, squares _have_ edges, so a `Square` class could have _members_ which are of an `Edge` type class. But `Edges` aren't squares, so `Edge` shouldn't derive from `Square` (or vice versa). 
    - This is called *composition* when the lifetime of the component is controlled by the class, and *aggregation* when the the component has an independent lifetime. 
        - A class representing a room has walls, which don't exist independently of the room and so can be represented using composition. The walls could be represented using member variables of type Wall, or pointer to Walls, possibly in a container. 
        - A room can also have a table, which could be moved to another room or thrown away, and hence exists independently of the room and can be represented using aggregation. There should be a pointer to an object of type Table, and some means to check that the Table is still in scope. 
- Inheritance is only for when you want a class to _be_ a kind of another class. 
- A mini-cooper **is** a type of car, so the class `MiniCooper` can **inherit** from the class `Car`. 
- A `Car` **has** an `Engine`, so the `Car` class should have a **member** of type (or pointer to type) `Engine`. 

## What is Inherited? 

- If we simply define a sub-class as inheriting from a base class, then it will inherit all of the member functions and variables which are not `private` from the base class definition. 

- Functions and member variables which exist in the base class don't need to be declared again in the derived class, unless you want to change how the function works in the derived class.

- A function which is defined in the base class and the derived class is said to be _overridden_ in the derived class. In this case when we call the function from an object of the derived class, the new function definition is used instead of the original definition in the base class. 

- The derived class has no access to private members of the base class, whether they are variables or functions. This does not mean that the derived class does not have these members: they are still part of the object's data because they are part of the base class. Private members of the base class could be indirectly manipulated, for example by public/protected functions defined in the base class (which are therefore available to the derived class) which act on or call private members of the base class. 

## Private, Protected, and Public Inheritance 

Like with access specifiers, a sub-class can inherit from a base class in three different ways. These kinds of inheritance have to do with how the sub-class controls access to members which it inherits from the base class. 

- `public`: Public inheritance is the most common form. When using public inheritance the access specifiers for members of the base class that are available to the derived class remain unchanged in the derived class. (Public members and protected members are still public and protected respectively; remember that private members in the base class are not available to the derived class.)
- `protected`: Protected inheritance converts public members of the base class to protected members of the derived class, so if you create an object of the derived type these member variables / functions will not be available outside the class unless it is converted to an instance of the base class. Protected members of the base class remain protected in the derived class. 
- `private`: Private inheritance converts public and protected members in the base class to private members in the derived class. 

## How is a Derived Object Created and Destroyed?

When a derived object is created:

1. The base class constructor is called. 
    - You can specify the base constructor to use for a given derived class constructor by writing a colon (`:`) followed by the base constructor you wish to call, e.g. `SubClass() : BaseClass(0, 0) { ... }`
    - If you do not specify a base constructor explicitly, e.g. `SubClass(){ ... }`, then the default constructor will be used. (If no default constructor for the base class exists you will get a compiler error.)
2. The derived class constructor is called second. 

When a derived object is destroyed:

1. The derived class destructor is called first. 
2. The base class destructor is called second. 
    - You can specify the base class destructor that you wish to use in the same way as the constructor.
    - As always, you must be extremely careful if you are doing any manual memory management. Memory must be freed, but must only be freed once. Don't free the same memory in the destructor of both the derived class and the destructor of the base class!

You can observe the creation and destruction of objects of base and derived classes by writing output in their constructor/destructor functions. 

## Overriding Inherited Functions

Unlike the constructor and destructor, most functions can be completely overridden by the base class. Calling the function in the derived class will not make any calls to the same function in the base class - the functionality is completely replaced. This is straight-forward to do: if we implement a function with the same name and signature as the base class (same type, name, number of arguments, and types of arguments) then this function will "override" the definition that would be inherited from the base class. 

Function overriding is fundamental to this polymorphic style of programming because this is what allows each sub-class to behave uniquely when placed in the same context.

## Polymorphism 

Polymorphism is the ability to use multiple types in the same context in our program; in order to achieve this we must only access the common properties of those types through some shared interface. The most common way to do this is to define a base class which defines the necessary common properties, and then have sub-classes which inherit from the base class which represent different kinds of objects which can implement this interface. This is caled *sub-type polymorphism*, and is one of the most common forms of polymorphism. 

By exploring polymorphism we can also understand the behaviour, and some of the limitations, of the straightforward model of inheritence that we have used so far. 

Let's assume that we have some class `Shape`, and derived classes `Circle` and `Square`.

```cpp
class Shape
{
    protected:
    Shape(){}

    public:
    Shape(double P, double A)
    {
        perimeter = P;
        area = A;
    }

    double getArea()
    {
        return area;
    }

    double getPerimeter()
    {
        return perimeter;
    }

    void printInfo()
    {
        cout << "Shape; Area = " << area << " m^2, Perimeter = " << perimeter << "m." << endl;
    }

    protected:
    double perimeter;
    double area;
};

class Circle : public Shape
{
    public:
    Circle(double r) : radius(r)
    {
        perimeter = 2 * M_PI * radius;
        area = M_PI * radius * radius;
    }

    void printInfo()
    {
        cout << "Circle; Radius = " << radius << "m, Area = " << area << " m^2, Perimeter = " << perimeter << "m." << endl;
    }

    protected:
    double radius;
};

class Square : public Shape
{
    public:
    Square(double w) : width(w)
    {
        perimeter = 4 * width;
        area = width * width;
    }

    void printInfo()
    {
        cout << "Square; Width = " << width << "m, Area = " << area << " m^2, Perimeter = " << perimeter << "m." << endl;
    }

    protected:
    double width;
};
```

- The `Shape` class has two functions: one to get the area (`getArea`) and one to get the perimeter (`getPerimeter`). 
    - These simply return member variables which store the area and perimeter of the shape, since there is no general formula for calculating the area or perimeter of an arbitrary shape. 
- The area and perimeter are set in the constructor of the `Shape` class. 
    - `Shape` also has a default constructor with no parameters. This is `protected` since it is used by the derived classes (which set the `area` and `perimeter` themselves) but can't be used outside of the class or derived classes: this means that we can't instantiate an object of type `Shape` using this constructor i.e. we cannot create a `Shape` with no area or perimeter. 
- The `Circle` and `Square` set the area / perimeter appropriately in their own constructors based on their relevant dimensions. 
    - `M_PI` is a constant defined in the header `<cmath>`.
- We also have a `printInfo` method which displays information about the shape to the terminal. This is overridden in the derived classes to display specialised information for each shape. 

Now let's say that we want to have a list of shapes, in the form of a vector, and get the area for each one. 

```cpp
void PrintShapeArea(Shape shape)
{
    cout << shape.getArea() << endl;
}

int main()
{
    Circle C = Circle(5.9);
    Square S = Square(3.1);

    PrintShapeArea(C);
    PrintShapeArea(S);
}
```

- When a `Circle` or `Square` is passed into `PrintShapeArea`, it is cast to a `Shape` type (the base class).
- It will lose any additional information or methods associated with the derived class. 
- The `Circle` and the `Square` both have access to the `perimeter` and `area` member variables, as well as their respective "getters". 
- The correct area will reported because the `area` member variable is set in the constructor, and the derived constructor has been called when the object was instantiated. 

Whenever we use a derived class in place of a base class, we implicitly cast to the base type and therefore can lose important information and behaviour defined in the derived class. In this example, we have separate `printInfo` functions for each of our classes. We run into a problem if we want to print this information for a list of `Shape` objects containing both `Circle` and `Square` objects. 

```cpp
void GetShapeInfo(Shape shape)
{
    shape.printInfo();
}

int main()
{
    Circle C = Circle(5.9);
    Square S = Square(3.1);

    C.printInfo();
    S.printInfo();

    std::cout << std::endl;

    GetShapeInfo(C);
    GetShapeInfo(S);
}
```

This will result in:

```
Circle; Radius = 5.9m, Area = 109.303 m^2, Perimeter = 37.052m.
Square; Width = 3.1m, Area = 9.61 m^2, Perimeter = 12.4m.

Shape; Area = 109.303 m^2, Perimeter = 37.052m.
Shape; Area = 9.61 m^2, Perimeter = 12.4m.
```

- When we call `printInfo()` from the derived class objects directly, we get their detailed information including the type of shape and the radius or width. 
- When we do the same on our objects within our vector, we only have access to the base class, and therefore we call the base class version of this method. 

In this case we have lost our specialised functionality for our derived classes when placed in a polymorphic context! In order for polymorphism to be really useful in C++, we need a way to retain the overridden functions for the derived classes, even when we are treating them in the more generalised context of a function or container which takes their base class. 

We shall see in the next section how we can make use of polymorphism whilst still accessing the functions of the derived class! 

## Virtual Functions 

Our current method of overriding and calling functions in the way described above is clearly insufficient in many cases where we want to use an object of a derived class in a piece of code which deals with the base class. Take for example a function that takes an argument of base type `Shape`:

- We often don't want to pass our derived class by value: this will attempt to copy the object into a new object of type `Shape`, so any overrides will be lost. 
- We should instead pass our argument by reference (or as a pointer, which we'll discuss in a later week). This will avoid the copying into a fresh object and instead will just pass the address in memory where the object we want to pass is stored. However, the function itself will still be treating the object as being of type `Shape` and hence will call the `Shape` versions of any functions. 

We can solve this problem by declaring a member function `virtual` in the base class. In this case, the function is accessed in a different way to normal. Function definitions have addresses, and normally when a member function of a class is called the definition of that function for that is just looked up. So if we are using a `Shape &` reference to an object, even if that object was created as type `Circle`, we will still look up the definition of any functions for `Shape`, since that's the class that we're using. For virtual functions however, each object will store the address of the definition of the function as part of its data (this data is called a "virtual table"). If the object is created as an instance of the base class, this will be the address of the base function, but if the object is created as an instance of a derived class, then this will be the address of the derived function. When we call the function on the object, it will execute the function at the address stored in the virtual table, which is individual to the instance of the object, rather than using an address which applies to the whole class. This means it doesn't matter if we are using a `Shape &` reference or `Circle &`, it will still used the derived function for the class `Circle` because that was the address put into the virtual table when the object was created. This is also why **passing a reference (or pointer) is necessary for this to work**. If we pass by value we will create a _new_ object of type `Shape`, and because it is of type `Shape` the new object's virtual table will link to the `Shape` implementation. If we pass a _reference_, then the function will instead look at the memory location of the original object, and therefore look in the original object's virtual table, and thus find the implementation for the derived class. 

Virtual functions open up fully polymorphic behaviour for our classes, and are important whenever a object of a derived class might be treated as a member of a base class, including:

- Passing objects of derived class to functions which take objects of base class (by reference or pointer).
- Defining a container of objects which can be of different derived classes by declaring a container using the base class. 
    - We will return to this technique later when we discuss pointers, you cannot have a container, such as `vector`, of references. Nevertheless it is good to be aware of this use case now as it is a very common way for polymorphism to come in handy! 

**N.B.** Special consideration should be given to _virtual destructors_. **If your class is inherited from, the destructor should usually be virtual.** We can point to an object of the derived class using a pointer of the type of `Base *`. If we `delete` this base pointer to free the memory then _only the base class destructor will be called_, and anything that needs to be cleaned up by the derived destructor will not happen. If the destructor is virtual, then the derived destructor will be called (which also calls the base destructor), and so any necessary clean up will happen. If you use _Smart Pointers_ to initialise your object then the correct (derived) destructor should be used even if the base destructor is not virtual. 

## Abstract Classes

Abstract classes are special cases of classes which have _virtual methods with no implementation_. Such functions are called **pure, virtual functions**. Such classes are abstract in the sense that they cannot be instantiated: we cannot create an object which is an instance of an abstract class because it has undefined functions and therefore the object to be instantiated is not fully defined. We can only instantiate objects of _derived classes_ which have implemented _all_ missing functionality. 

- Abstract classes can be used when we want to define a **type** of object where any instance must be one of a set of **concrete sub-types**.
    - It often useful for modelling abstract concepts defined by some shared properties. For example, many different things are animals, but every animal alive is a specific species, i.e. sub-type, of animal. So we don't want to be able to instantiate an "animal" type object without declaring its species as well: the derived type is concrete and can exist, but the base type is abstract and merely denotes membership of a broader type class. 
- Abstract classes are any class which has at least one pure, virtual function
    - A function declared pure by setting it `= 0` in the definition 
    - e.g. `virtual int myPureVirtualFunction(int a, int b) = 0;`
- Abstract classes allow us to model interfaces which have no default (base) implementation but which may have many possible implementations. 
- Although abstract classes cannot be instantiated on their own, they still have constructors and destructors, which are called in the same way as other base classes. These can be used to set or clean up data present in the definition of the abstract class. 
- Destructors for abstract classes should be virtual, since instances of abstract classes are _always_ derived classes and so we should make sure that the derived destructor is always called.  

Let's return to our `Shape` class example, which defines shapes as a class of objects which have an area and a perimeter which can be calculated. `Shape` is a good candidate for an abstract class, because area and perimeter have no meaningful implementation until the form of the shape is specified, and thus there is no reasonable base class implementation, but we want this functionality to be available in any actual shapes which are created. We can then create derived classes for triangles, circles, and squares which override these pure virtual methods, and therefore can be instantiated. Now we can treat instantiations of each of the derived classes as objects of type Shape, and pass them to the same functions and containers, without any risk that an invalid and meaningless Shape base object will be created. 

Here's a new definition of `Shape`, `Circle`, and `Square` that makes `Shape` and abstract class, and replaces the `area` and `perimeter` member variables with functions that calculate these properties instead. 

```cpp
class Shape
{
    public:
    Shape()
    {
    }

    virtual double getArea() = 0;

    virtual double getPerimeter() = 0;

    virtual void printInfo() = 0;
};

class Circle : public Shape
{
    public:
    Circle(double r) : radius(r){}

    void printInfo()
    {
        cout << "Circle; Radius = " << m_radius << "m, Area = " << m_area << " m^2, Perimeter = " << m_perimeter << "m." << endl;
    }

    double getArea()
    {
        return M_PI * radius * radius;
    }

    double getPerimeter()
    {
        return 2 * M_PI * radius;
    }

    protected:
    double radius;
};

class Square : public Shape
{
    public:
    Square(double w) : width(w){}

    double getArea()
    {
        return width * width;
    }

    double getPerimeter()
    {
        return 4 * width;
    }

    void printInfo()
    {
        cout << "Square; Width = " << width << "m, Area = " << area << " m^2, Perimeter = " << perimeter << "m." << endl;
    }

    protected:
    double width;
};
```

- We can no longer make objects of type `Shape`, only `Circle` or `Square`. 
- We can however have pointers (smart pointer or raw pointers) or references to `Shape`, which will use the derived versions of `getArea`, `getPerimeter`, and `printInfo` for each object depending on whether it was created as a `Circle` or `Square`. 
- The use of virtual functions makes this version more polymorphic than the previous one. 
- The use of pure virtual functions means that the `Shape` class more closely corresponds to our abstract notion of a shape as being something that we can't implement without more information. 
- Note that we don't have to design the class so that we re-calculate the area and perimeter every time we call `getArea` and `getPerimeter`; we could store them in member variables like in our previous example. Think about the pros and cons of these two approaches! 

