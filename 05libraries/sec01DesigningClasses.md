---
title: Designing Classes and Code 
---

Estimated Reading Time: 40 Minutes

# Designing Classes & Code

Classes in C++ are extremely flexible, and there are many different ways that we could design classes for different purposes. We've seen how we can use inheritance and composition to describe relationships between classes and objects, and how to control how information is accessed through access specifiers, but the ways in which we put these concepts to work can have a major impact on our code. These kinds of decisions rarely have a right and wrong answer, and there is a great deal of discussion in the C++ community over what constitutes good practice, but there are some techniques and principles that you will likely find yourself using repeatedly if you work with C++ in the future and are useful to learn about now. 

Good class design - and code design in general - is tailored towards the goals and priorities of a given project. In this course we will be focussing on programming for academic research, but in your career you may find yourself programming in many different contexts, and it is instructive to think about how differing priorities can lead to different design choices. 

## How Do We Think About Code Quality?

Programming is not just about producing code that works or passes tests. If code is not also usable, it will not be used. If code is not also maintainable, it will become unusable. There are many aspects to code quality, the importance of which depends on the intended use of the code. A very dry list of code quality measures can be found in [Steve McConnell's *Code Complete*][code-complete]:

- External qualities
  - Correctness
  - Accuracy
  - Reliability
  - Robustness
  - Efficiency
  - Adaptability
  - Usability
- Internal qualities 
  - Maintainability
  - Flexibility
  - Portability
  - Reusability
  - Readability
  - Testability
  - Understandability

McConnell splits the list into external qualities, which are important to the *user* of the code, and internal qualities, which are important to the programmers who contribute to the code. Let's look at each of these in detail.

### **External Qualities**

**Correctness**

Does the program do what it is meant to?

**Accuracy**

Are the results close enough to what I need?

**Reliability**

Are the results the same every time the program is run?

**Robustness**

Does the program handle unexpected inputs correctly?

**Efficiency**

Is the program fast enough?

**Adaptability**

Can I extend the program to do something similar but unintended?

**Usability**

Is the program easy to setup and use?

### **Internal qualities**

**Maintainability**

Are bugs easy to find and to fix?

**Flexibility**

Can I easily add new features to the code?

**Portability**

Am I able to run the program on many different architectures and operating systems?

**Reusability**

Can I use parts of the code in many different places?

**Testability**

Is the code written and designed in such a way that it's easy to test?

**Readability**

How much time do I spend trying to read the code on a surface-level?

**Understandability**

How much time do I spend trying to understand the code?

Some of these qualities are related; readability and understandability are very much entwined, as are correctness and accuracy. Being able to think about code from the perspectives of all these qualities gives you, as a programmer, a much better understanding of what it means to write good code. In saying that, codes designed for different purposes may have very different priorities when it comes to quality. For example, the software controlling a nuclear power station must be extremely correct, robust and reliable, but it may not need to be particularly portable, adaptable or even efficient. On the other hand, a game on you phone can prioritise usability and portability at the expense of some maintainability, robustness and even correctness (if it crashes or glitches, does it matter *that* much?).


## Some Guiding Principles 

In general when programming we want our code to be _modular_ in some way: we want our code to exist in independent chunks, each of which achieves a simple task, which can then be combined to perform more complex tasks. Writing functions is a good example of this, where each function generally calculates a single thing, but combining functions can process data in very complex ways. We can approach classes in a similar way, breaking structures into smaller classes which represent individual things and building more complex structures from these components through processes like composition or inheritance. This leads to more flexible representations that allow us to make efficient changes to our code. 

Below are some commonly recurring ideas in software design that you may want to consider. Bear in mind however, that there are no strict rules governing "good" or "bad" code design, and therefore no rule is applied 100% of the time. Make sure you always keep the goals of your code in mind when making choices about how to write it. 

### **Program To Interfaces**

We've seen when discussing inheritance that inferfaces can be created using abstract classes or templates. Interfaces are a key aspect of flexible, extensible class design, as they specify the expected functionality that a class will need without restricting the specific class. This means that if we program to an interface we can change the implementation of that component without changing any of the code for the parts of the program which use it, making updating our code much more efficient and flexible. This is generally a good property of programs that we should strive for when possible.

Suppose, for example, that our program needs to load and display an image. There exists many different file formats for images, which all represent the images in different ways. In a bitmap image the colour values are stored for every pixel individually, and the image can be drawn by going through it pixel by pixel. A jpeg image on the other hand uses a special compression format that breaks the image into blocks (8 pixels by 8 pixels) and stores _fourier components_ for those blocks. In order for the image to be reconstructed and displayed, we have to reconstruct each block by multiplying those fourier components by the basis functions. The functionality for the end user is the same either way: load an image, and display it. Code that the user writes shouldn't have to care what the format of the image is in order to be able to use it, and this is where an interface becomes useful. If we have an `ImageDisplay` interface which promises a `load` and `display` function will be available, then we can write code which works with arbitrary objects which inherit from this class. We can then choose which implementation of the `ImageDisplay` interface to use based on the kind of file we are loading just at one single point (when the object is created), and the rest of the code is completely agnostic about that choice of implementation. 

Different object-oriented languages have different approach to implementing the concepts of interfaces. C++ has no explicit "interface" construct, but interfaces are usually implemented by abstract classes. 
- An interface is a guarantee that an object will implement a certain set of functionality.
- The abstract class that defines an interface should not implement that functionality itself.
- The functionality defined in an interface should be minimal to describe what is needed from objects of this kind. 

### **KISS: Keep It Simple, Stupid**

The principle of _KISS_ is straight-forward: try to keep your programs simple and understandable. Don't obfuscate your code with complex structures that you don't need or use, but keep things direct and make sure the structure of your code reflects the concepts that you are trying to model. 

### **DRY: Don't Repeat Yourself**

The basic principle here is to organise your code into re-usable blocks rather than rewriting the same functionality repeatedly. If you have some calculation that appears multiple times throughout your code, you can place that calculation inside a function which is called in different places to avoid repeating that logic. 

## RAII: Resource Acquisition Is Initialisation

The concept of RAII is of great importance in a language like C++ which allows you to manage memory manually. This pattern is designed to make your classes memory safe, and generally entails two key principles:

- Memory allocated by your class should be allocated in the constructor.
- Memory allocated by your class should be de-allocated in the destructor. 

If the constructor fails to allocate the resources required for the class, then it should throw an exception. Any resources already allocated by the class before reaching the exception should be deallocated. 

The goal is to guarantee the following:

- Resources that are required by the object exist for the full lifetime of the object. This will prevent invalid memory access attempts.
- Resources that are allocated by the object do not exist for longer than the object itself. This will prevent memory leaks. 

Since it's good practice to use smart pointers for any pointers which actually own data (and therefore we should not need to manually make calls to `delete` in our destructor), the main times when we need to be concerned with RAII are in dealing with opening and reading or writing resources such as files. However, RAII can also be very useful for interfacing with C libraries which deal with raw pointers and which have specialised methods for creating and freeing them (rather than using `new` and `delete`); it can also often be easier to deal with C-style arrays rather than vectors when programming with MPI or interfacing to other devices like GPUs. 

RAII typically means wrapping these resources that you want to use in some class: rather than accessing a file directly in a function, which could be interrupted by an exception before it can close the file, wrap the file in a class which will automatically close the file in the destructor if the object goes out of scope. Then use that class in your function to access your file. If something goes wrong and an exception is thrown, your file will be closed when the stack unwinds and the file wrapper object is deleted. 

### RAII and Copying

Special care needs to be taken when objects that implement the RAII pattern are allowed to be copied. C++ will, where possible, implement a _default_ [copy constructor](https://en.cppreference.com/w/cpp/language/copy_constructor), which allows the object to be copied, e.g. 

```cpp
MyObj obj1;
MyObj obj2 = obj1;  // calls copy constructor to build obj2 by copying data from obj1
```

The trouble with copies comes when we have classes which contain resources like raw pointers or file handles that need to be deleted or closed.

```cpp
class MyObj
{
public:
    MyObj()
    {
        p = new int(5);
    }

    ~MyObj()
    {
        delete p;
    }

private:
    int *p;
};
```
- This class very responsibly places the allocation for the pointer in the constructor and the deallocation in the destructor, so creating a `MyObj` and letting it go out of scope will not cause any leaks.

The problem with the raw pointer is that the default copy will simply copy the pointer across. This means that **both** `obj1` and `obj2` will contain a pointer **to the same address**, and consequenctly **both destructors will attempt to free it**. This is a double free error and will cause our program to crash! We have failed to properly model _ownership_ of the resources in the case of the copy: we must always know which objects own what resources. 

When smart pointers are not appropriate, we can control this copy behaviour by overriding or disabling the copy constructor. 

#### Overriding the Copy Constructor

The copy constructor for a given type looks like this:

```cpp
class MyObj
{
public:
    // copy constructor
    MyObj(const MyObj &other)
    {
        ...
    }

...
```
- It takes a (possibly `const`) _reference_ to an object of the same class as its argument. It's usually a good idea to make this a `const` reference since you probably don't want your copy operation to be able to alter the original object!
- It can take other parameters if you want **but** they must have default values supplied in the argument list. E.g. `MyObj(const MyObj &other, int i=2){...}`.

We can override this to make a deep copy by having the new object's pointer point to a different memory location, and instead copy the _data_ that the first object points to into the new location as well. 

```cpp
class MyObj
{
public:
    // copy constructor
    MyObj(const MyObj &other)
    {
        p = new int(*other.p);
    }

...
```

Note that this deep copy means that the data that these two objects point to is now independent: changing one won't change the other because they are looking at different addresses. 

#### Disabling the Copy Constructor

We can prevent copying entirely by disabling the copy constructor. 

```cpp
class MyObj
{
public:
    // copy constructor
    MyObj(const MyObj &other) = delete;

...
```
This makes it a compilation error to try to copy the object, and therefore the our code `MyObj obj2 = obj1;` won't compile at all. 

There are more approaches that one can take to this problem depending on exactly what ownership behaviour you want, just **always remember to consider ownership when implementing classes with RAII**. 

## Decoupling Code with Abstract Classes & Dependency Injection 

Dependency injection is a commonly used technique to make a pair of classes which depend on one another _loosely coupled_, i.e. to make changes to one class as independent of the other class as possible. 

Consider for example the case where we have one class which contains an instance of another. In this case, a `Simulation` class which contains a simple `Data` class. 

```cpp
class Data
{
    public:
    void print()
    {
        for(auto x: data)
        {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

    private:
    vector<int> data;
};

class Simulation
{
    public:
    Simulation()
    {
        data = std::unique_ptr<Data>(new Data());
    }

    void printData()
    {
        data->print();
    }

    private:
        std::unique_ptr<Data> data;
};
```
- The definition of class `Simulation` is dependent on the definition of class `Data`. 
- The constructor for `Simulation` calls the constructor of `Data` directly; if the constructor of `Data` changes (because we have changed something about our data representation) then the class `Simulation` must also be changed. 
- The class `Data` may develop and contain functionality that is irrelevant to what `Simulation` needs. 

Dependency injection breaks the coupling between these classes by contructing the component (`Data`) _separately_, and passing it into the constructor for the class of object that will hold it (`Simulation`). This means that if we change the way that the `Data` class is constructed, we don't have to make any changes to the `Simulation` class. 

```cpp
class Simulation
{
    Simulation(unique_ptr<Data> &inData)
    {
        data = std::move(inData);
    }

    void printData()
    {
        data->print();
    }

    private:
        std::unique_ptr<Data> data;
};
```

- Note that the `Simulation` class now does not call the constructor for the `data` object: the `Data` implementation can change completely as long as it still implements the `print` method, which is the only thing that we need from it in this example. The `Simulation` class is now _decoupled_ from any elements of the `Data` class that it does not directly need to know about and use. 

We can also add a setter function using a similar appraoch. With this kind of structure we can create classes that allow components to be swapped out during the lifetime of the object, something that will be very important for the next pattern that we look at.

```cpp
class Simulation
{
    public:
    Simulation(unique_ptr<Data> &inData)
    {
        data = std::move(inData);
    }

    void setData(unique_ptr<Data> &inData)
    {
        data = std::move(inData);
    }

    void printData()
    {
        data->print();
    }

    private:
        std::unique_ptr<Data> data;
};
```
- If we have two data sets `dataSet1` and `dataSet2` we can now change the data that the `Simulation` object looks at runtime without creating a new `Simulation` object. 

Using setters to implement dependency injection should not lead you to neglect your constructors! It's generally best if you always **make sure that objects are constructed in a valid state**. In this example, we might consider any object where the `data` pointer is null to be invalid. In order to avoid a partially constructed object, we need to make sure the dependency injection is implemented in the constructor (and optionally the setter), and always checks for null pointers. 

```cpp
class Simulation
{
    public:
    Simulation(unique_ptr<Data> &inData)
    {
        if(!inData)
        {
            throw std::runtime_error("Simulation error: data set pointer cannot be null.")
        }
        data = std::move(inData);
    }

    void setData(unique_ptr<Data> &inData)
    {
        if(!inData)
        {
            throw std::runtime_error("Simulation error: data set pointer cannot be null.")
        }
        data = std::move(inData);
    }

    void printData()
    {
        data->print();
    }

    private:
        std::unique_ptr<Data> data;
};
```

## Example: Strategy Pattern

The _strategy pattern_ takes advantage of polymorphic behaviour to provide different solutions or representations for problems at runtime. 

Often the best solution to use for a particular problem will depend on the details of that problem: some sorting algorithms are faster on shorter lists while other are faster on longer lists, or some integrators might be more accurate and efficient when integrating slowly changing functions but others work better for oscillating functions. If we have a class which needs to have a component which achieves a task like this, then rather than having multiple classes with different concrete implementations of these algorithms built in, we can have a single class with a pointer to an abstract base class e.g. a sorter, or an integrator. We can then have different sub-classes of our abstract class that implement the different solutions to our problem, and we can pass different solutions into our main class, or even change solutions throughout the runtime based on considerations that we can't know at compile time (e.g. the length of a list that needs to be sorted). Let's continue our example from dependency injection, but now let's add an abstract base class for the `Data`. 


```cpp
class AbstractSimData
{
    public:
    virtual void print() = 0;
};

class Data : public AbstractSimData
{
    public:
    void print()
    {
        for(auto x: data)
        {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

    private:
    vector<int> data;
};
```
- `AbstractSimData` is an abstract class, because its function `print` is not implemented. It defines the interface that any data class that wants to be used with the `Simulation` class would need to implement. This interface should be kept to the minimum required.
- `print` is _pure_ and _virtual_ which means that it will always be overridden by a derived class. This defines a "contract": a set of functionality that anything which inherits from this abstract class _must_ implement. We can use such abstract classes to define minimal functionality required by other classes: this is sometimes referred to as an "interface". 
    - Interfaces are a core language feature of some other languages like Java and C#, but are not explicitly implemented in C++. 
    - In C++ we generally implement interfaces using abstract classes containing only pure virtual functions and variables. 

With this in place, we can define multiple classes which inherit from `AbstractSimData`, and which can then store data in different ways (e.g. maps instead of vectors and so on), or implement functions differently. 

```cpp
class Simulation
{
    public:
    Simulation(unique_ptr<AbstractSimData> &inData)
    {
        data = std::move(inData);
    }

    void setData(unique_ptr<AbstractSimData> &inData)
    {
        data = std::move(inData);
    }

    void printData()
    {
        data->print();
    }

    private:
        std::unique_ptr<AbstractSimData> data;
};
```

- `Simulation` doesn't care _how_ the `data` component implements `print`, it only needs to know that it _can_. The abstract base class has captured everything that is required by the `Simulation` class. 
- With the setter in place we can change the approach that we are taking whenever we need to. 

## Example: Factory Pattern 

When dealing with abstract classes it is sometimes useful to be able to make objects of different sub-classes depending on runtime considerations. In this case, we can define another class or method, sometimes known as a "factory", which returns something of the base type. Common examples might be selecting an approach using a runtime flag, or changing approaches based on the size of the problem. 

A factory can be implemented as a class, but often the simplest approach (in C++) is to just have a factory function like the one below. 

```cpp
std::unique_ptr<AbstractDataManager> DataManagerFactory(std::vector &v)
{
    if(v.size() < 1000)
    {
        return std::make_unique<ShortDataManager>(v);
    }
    else
    {
        return std::make_unique<LargeDataManager>(v);
    }
}
```

- `AbstractDataManager` is an abstract base class
- `ShortDataManager` and `LargeDataManager` are two concrete classes which inherit from `AbstractDataManager`, and are optimised for dealing with data on different scales. 
- The factory uses the size of the data being passed in to make a decision about the approach that is going to be taken. The size of the vector is only known at runtime. 

## Implementing Multiple Interfaces 

 When working with interfaces which define minimal behaviour, it is possible that useful objects will need to implement more than one set of behaviour. This is easy to do when the sets of behaviour are nested i.e. we can model it with a chain of inheritance. (An `Undergraduate` is a type of `Student` which is a type of `UniversityMember`.) Things can be slightly more complicated when an object implements two different functionalities which are independent of one another, for example a `StudentTeachingAssistant` is a `Student` and an `Employee`, but a `Student` is not a type of `Employee` (and vice versa). 
 
 In this case we would need to implement two interfaces independently, which can be done using multiple inheritance. Multiple inheritance is a complex topic in C++ that goes beyond implementing multiple abstract interfaces, so you should think carefully about whether, and how, you use it. A good rule of thumb is to avoid it where you can (and limit inheritance to tree-like structures), and if you _do_ need to use it then limit yourself to inheritance from abstract classes with no explicit implementation or data members. 

As an example, let's say we have a computer system that models people in a university and it has two classes `Student` and `Employee` to represent roles which can be taken by people in the university. We'll have some systems that process employees, and some that deal with students. There will likely be multiple types of students and employees, which we can fit seamlessly into these systems by having derived classes which inherit from `Student` or `Employee`. Now say we want a class to model a student teaching assistant - these are both students _and_ employees, and should be able to function in parts of the system that deal with either. In this case we need a class that is recognisable as both a `Student` and an `Employee` in our program. We can do this by declaring multiple inheritance:
```cpp
class StudentTA : public Student, public Employee {};
```
- We can declare a class to inherit from as many classes as we like
- We need to declare `public`, `private`, or `protected` for each class we inherit from. 

There are some special problems that we need to look out for when working with multiple inheritance. 
- If more than one base class contains a member with the same name, then calling for that member from the derived class will be ambiguous and result in a compiler error. For example, if `Student` and `Employee` both have a `department` member variable (or function), calling `myTA.department` will be ambiguous unless it is overridden by the `StudentTA` class. 
    - Remember that inherited classes contain copies of the base classes that they inherit from. 
    - This means that if the variable `myTA` is of type `StudentTA` then it will actually have two different `department` variables: `Student::department` and `Employee::department`. These are independent of one another. 
    - These can be explicitly accessed using e.g. `myTA.Student::department`. (Note that this can be done in derived classes in general, if you want to explicitly access the member of a base class even without multiple inheritance.)
    - **N.B.** This is not a problem when working with interfaces since this can resolve the ambiguity. For example, if I have a function such as `calculatePayroll(Employee &person)` which accesses the `department` variable, then because the person is being interpreted as being of type `Employee` in this context, calling `person.department` will unambiguously look for the department associated with the `Employee` class. This is because an object being treated as an `Employee` can only have access to the `Employee` members, and so can't know about any members from the `Student` class.
    - Try not to duplicate functionality like `department` in multiple classes. If closely related classes implement the same functionality, it could be that this is better handled by a super-class from which they inherit e.g. `UniversityMember`.
- Special problems arise from the "Diamond Problem": this is where a class inherits from two classes, which in turn both inherit from the same class. For example, if we have a `UniversityMember` class, and `Employee` and `Student` both inherit from `UniversityMember`, and then `StudentTA` inherits from both `Employee` and `Student`. 
    - In this case our `StudentTA` object contains both `Employee` and `Student` objects. 
    - The `Employee` object contains a `UniversityMember` object.
    - The `Student` object contains a `UniversityMember` object as well. 
    - This means our `StudentTA` has two versions of all of the `UniversityMember` member variables and functions! These cannot be made unambiguous because they originate from the same class, and thus we can't use the namespace to differentiate. 
    - We can solve this using _virtual inheritance_. If `Employee` and `Student` both use virtual inheritance from person (`class Employee : virtual public UniversityMember`) then there will only ever be one copy of the base class.


[code-complete]: https://learning.oreilly.com/library/view/code-complete-second/0735619670/