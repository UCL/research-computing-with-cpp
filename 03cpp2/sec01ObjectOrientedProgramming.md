---
title: Object Oriented Programming
---

Estimated Reading Time: 15 minutes

# Object Oriented Programming (OOP) in C++

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