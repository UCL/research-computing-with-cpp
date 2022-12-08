# Inheritance

Inheritance is one of the most important concepts in C++ classes, which brings a great deal of flexibility to us as programmers. A class defines a type of object, and a class which inherits from it defines a sub-type of that object. For example, we might have a class which represents shapes, and sub-classes which represent squares, circles, and triangles. Each of these are shapes, and so should be able to be used in any context simply requires a shape, but each will have slightly different data needed to define it and different implementations of functions to calculate its perimeter or area. 

## When Should Inheritance Be Used?

- Inheritance should be used only when you want to declare that one class is a sub-type of another class. Essentially `B` may inherit from `A` is `B` is a kind of `A`.
- A common example is that the classes `Circle` and `Square` may both derive from the class `Shape`. But neither `Circle` nor `Square` should inherit from one another! 
- The **Liskov Substitution Principle** is one good guiding principle. 
    - If `B` is a sub-type of `A`, then replacing an object of type `A` with an object of type `B` should not break your program. 
    - In this case a `B` object can be considered a kind of `A` object, but not the other way around. 
    - Sub-types are more specific than base types. 
- Derived classes should generally _extend_ classes rather than restrict classes. Having a subclass that is simpler than the base class can cause problems if the object is substituted into a part of the program that expects a base class, as functionality expected for the base class may not be appropriately defined in the derived class. 
    - This is commonly referred to as the Square/Rectangle problem or the Circle/Ellipse Problem
    - In this case the square (or circle) is a special case of a rectangle (or ellipse) which is more restricted an contains less information, because it only requires one length to be defined instead of two. 
    - Functionality that manipulates the height and width of the rectangle individually don't make sense for a square, because it should only have one. 

## Composition: When **not** to use Inheritance

- Don't use inheritance if you want a class to _have_ an instance of another class as a component. 
    - It should be achieved by having a member variable of that type, or a pointer or reference to an object of that type. 
    - This is called *composition* when the lifetime of the component is controlled by the class, and *aggregation* when the the component has an independent lifetime. 
        - A class representing a room has walls, which don't exist independently of the room and so can be represented using composition. The walls could be represented using member variables of type Wall, or pointer to Walls, possibly in a container. 
        - A room can also have a table, which could be moved to another room or thrown away, and hence exists independently of the room and can be represented using aggregation. There should be a pointer to an object of type Table, and some means to check that the Table is still in scope. 
- Inheritance is only for when you want a class to _be_ a kind of another class. 
- A mini-cooper **is** a type of car, so the class `MiniCooper` can **inherit** from the class `Car`. 
- A `Car` **has** an `Engine`, so the `Car` class should have a **member** of type (or pointer to type) `Engine`. 

## What is Inherited? 

- If we simply define a sub-class as inheriting from a base class, then it will inherit all of the member functions and variables which are not `private` from the base class definition. 

- Functions and member variables which exist in base class don't need to be declared again in the derived class. 

- A function which is defined in the base class and the derived class is said to be _overridden_. In this case when we call the function from an object of the derived class, the new function definition is used instead of the original definition in the base class. 

- The derived class has no access to private members of the base class, whether they are variables or functions. This does not mean that the derived class does not have these members: they are still part of the object's data because they are part of the base class. Private members of the base class could be indirectly manipulated, for example by public/protected functions defined in the base class (which are therefore available to the derived class) which act on or call private members of the base class (which are not directly available). 

## Private, Protected, and Public Inheritance 

Like with access specifiers, a sub-class can inherit from a base class in three different ways. These kinds of inheritance have to do with how the sub-class controls access to members which it inherits from the base class. 

- `public`: Public inheritance is the most common form. When using public inheritance the access specifiers for members of the base class that are available to the derived class remain unchanged in the derived class. (Public members and protected members are still public and protected respectively; remember that private members in the base class are not available to the derived class.)
- `protected`: Protected inheritance converts public members of the base class to protected members of the derived class, so if you create an object of the derived type these member variables / functions will no longer be available outside the class unless it is converted to a base class. Protected members of the base class remain protected in the derived class. 
- `private`: Private inheritance converts public and protected members in the base class to private members in the derived class. 

## How is a Derived Object Created and Destroyed?

When a derived object is created:
1. The base class constructor is called. 
    - You can specify the base constructor to use for a given derived class constructor by writing a colon (`:`) followed by the base constructor you wish to call, e.g. `SubClass() : BaseClass(0, 0) { ... }`
    - If you do not specify a base constructor explicitly, e.g. `SubClass(){ ... }`, then the default constructor will be used. (If no default constructor for the base class exists you will get a compiler error.)
2. The derived class constructor is called second. 

When a derived object is destroyed:
1. The derived class destructor is called first. 
2. The base class destructor is callled second. 
    - You can specify the base class destructor that you wish to use in the same as the constructor.
    - As always, you must be extremely careful if you are doing any manual memory management. Memory must be freed, but must only be freed once. Do free the same memory in the destructor of both the derived class and the destructor of the base class! 

You can explore the creation and destruction of objects of base and derived classes by writing ouput in their constructor/destructor functions. 

## Overriding Inherited Functions

Unlike the constructor and destructor, most functions can be completely overridden by the base class. This is straight-forward to do: if we implement a function with the same name and signature as the base class (same type, name, number of arguments, and types of arguments) then this function will "override" the definition that would be inherited from the base class. 

## Polymorphism 

One of the most important applications of class inheritance is to implement a kind of *polymorphism*. Polymorphism in programming refers to cases where a single interface can be used with multiple different types in one way or another. The most common way to do this is to define some base class which defines the necessary common properties, and then have sub-classes which inherit from the base class which represent different kinds of objects which can meet this interface. This is caled *sub-type polymorphism*, and is one of the most common forms of polymorphism. 

By exploring polymorphism we can also understand the behaviour, and some of the limitations, of the straightforward model of inheritence that we have used so far. 

Let's assume that we have some class `Shape`, and derived classes `Circle` and `Square`.

```cpp
class Shape
{
    public:
    Shape(double perimeter, double area)
    {
        m_perimeter = perimeter;
        m_area = area;
    }

    double getArea()
    {
        return m_area;
    }

    double getPerimeter()
    {
        return m_perimeter;
    }

    void printInfo()
    {
        cout << "Shape; Area = " << m_area << " m^2, Perimeter = " << m_perimeter << "m." << endl;
    }

    protected:
    double m_perimeter;
    double m_area;
};

class Circle : public Shape
{
    public:
    Circle(double radius) : Shape(2 * 3.14 * radius, 3.14 * radius * radius){}

    void printInfo()
    {
        cout << "Circle; Radius = " << m_radius << "m, Area = " << m_area << " m^2, Perimeter = " << m_perimeter << "m." << endl;
    }

    protected:
    double m_radius;
};

class Square : public Shape
{
    public:
    Square(double width) : Shape(width * 4, width * width), width(width){}

    void printInfo()
    {
        cout << "Square; Width = " << width << "m, Area = " << area << " m^2, Perimeter = " << perimeter << "m." << endl;
    }

    protected:
    double width;
};
```

Now let's say that we want to have a list of shapes, in the form of a vector, and get the area for each one. 

```cpp
void GetShapeAreas(vector<Shape> shapes)
{
    for(auto &shape : shapes)
    {
        cout << shape.getArea() << endl;
    }
}

int main()
{
    Circle C = Circle(5.9);
    Square S = Square(3.1);

    vector<Shape> shapes;
    shapes.push_back(C);
    shapes.push_back(S);

    GetShapeAreas(shapes);
}
```
- When a `Circle` or `Square` is placed into the `vector<Shape>` container, it is cast to a `Shape` (the base class).
- It will lose any additional information or methods associated with the derived class. 
- The `Circle` and the `Square` both have access to the `perimeter` and `area` member variables, as well as their respective "getters". 
- The correct area will reported because the `area` member variable is set in the constructor, and the derived constructor has been called when the object was instantiated. 

Whenever we use a derived class in place of a base class, we implicitly cast to the base type and therefore can lose important information and behaviour defined in the derived class. In this example, we have separate `printInfo` functions for each of our classes. We run into a problem if we want to print this information for a list of `Shape` objects containing both `Circle` and `Square` objects. 

```cpp
void GetShapeInfo(vector<Shape> shapes)
{
    for(auto &shape : shapes)
    {
        shape.printInfo();
    }
}

int main()
{
    Circle C = Circle(5.9);
    Square S = Square(3.1);

    vector<Shape> shapes;
    shapes.push_back(C);
    shapes.push_back(S);
    
    std::cout << std::endl;

    C.printInfo();
    S.printInfo();

    GetShapeInfo(shapes);
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

We shall see in the next section how we can make use of polymorphism whilst still accessing the functions of the derived class! 

## Virtual Functions 

There are some cases where overriding functions in the way described above is not sufficient. This is usually when we want to use an object of a derived class in a piece of code which deals with the base class. Take for example a function that takes an argument of type `Base`
- We often don't want to pass our derived class by value: this will attempt to copy the object into a new object of type `Base`, so any overrides will be lost. 
- We can pass our argument by reference or as a pointer. However, the function itself will still be treating the object as being of type `Base` and hence will call the `Base` versions of any functions. 

We can solve this problem by declaring a member function `virtual` in the base class. In this case, the function is accessed slightly differently to normal. Instead of just looking up the definition of the function for that class, the object will store the address of a function as part of its data (this data is called a "virtual table"). If the object is created as an instance of the base class, this will be the address of the base function, but if the object is created as an instance of a derived class, then this will be the address of the derived function. When we call the function on the object, it will execute the function at the address stored in the data. This means it doesn't matter if we are treating it as a `Base` or `Derived`, it will still used the derived function. 

Virtual functions are important whenever a object of a derived class might be treated as a member of a base class, including:
- Passing objects of derived class to functions which take objects of base class (by reference or pointer).
- Placing a pointer to an object of derived class in a container (such as `vector`) of pointers to the base class.

**N.B.** Special consideration should be given to _virtual destructors_. **If your class is inherited from, the destructor should be virtual.** We can point to an object of the derived class using a pointer of the type of `Base *`. If we `delete` this base pointer to free the memory then _only the base class destructor will be called_, and anything that needs to be cleaned up by the derived destructor will not happen. If the destructor is virtual, then the derived destructor will be called (which also calls the base destructor), and so any necessary clean up will happen. 

## Abstract Classes

Abstract classes are special cases of classes which have _virtual methods with no implementation_. They are abstract in the sense that they cannot be instantiated: we cannot create an object which is an instance of an abstract class because it has undefined functions and therefore the object to be instantiated is not fully defined. We can only instantiate objects of _derived classes_ which have implemented any missing functionality. 
- Allows us to model interfaces which have no default (base) implementation but which may have many possible implementations. 
- Abstract classes are any class which has at least one **pure, virtual function**
    - A function declared pure by setting it `= 0` in the definition 
    - `virtual int myPureVirtual(int a, int b) = 0;`
- Although abstract classes cannot be instantiated on their own, they still have constructors and destructors, which are called in the same way as other base classes. 
- Destructors for abstract classes should be virtual, since instances of abstract classes are _always_ derived classes and so we should make sure that the derived destructor is always called.  

We might, for example, want to create a class to represent shapes, all of which have some area or perimeter which can be calculated. This is a good candidate for an abstract class, because area and perimeter have no meaningful implementation until the form of the shape is specified, and thus there is no reasonable base class implementation, but we want this information to be available in any actual shapes which are created. We can then create derived classes for triangles, circles, and squares which override these pure virtual methods, and therefore can be instantiated. Now we can treat instantiations of each of the derived classes as objects of type Shape, and pass them to the same functions and containers, without any risk that an invalid and meaningless Shape base object will be created. 


