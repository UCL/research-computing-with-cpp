---
title: Designing Classes & Code 
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

In general when programming we want our code to _modular_ in some way: we want our code to exist in independent chunks which achieve a simple tasks, which can then be combined to perform more complex tasks. Writing functions is a good example of this, where each function generally calculates a single thing, but combining functions can process data in very complex ways. We can approach classes in a similar way, breaking structures into smaller classes which represent individual things and building more complex structures from these components through processes like composition or inheritance. This leads to more flexible representations that allow us to make efficient changes to our code. 

Below are some commonly recurring ideas in software design that you may want to consider. Bear in mind however, that there are no strict rules governing "good" or "bad" code design, and therefore no rule is applied 100% of the time. Make sure you always keep the goals of your code in mind when making choices about how to write it. 

### **Program To Interfaces**

We've seen when discussing inheritance that inferfaces can be created using abstract classes or templates. Interfaces are a key aspect of flexible, extensible class design, as they specify the expected functionality that a class will need without restricting the specific class. This means that if we program to an interface we can change the implementation of that component without changing any of the code for the parts of the program which use it, making updating our code much more efficient and flexible. This is generally a good property of programs that we should strive for when possible. 

Different object-oriented languages have different approach to implementing the concepts of interfaces. C++ has no explicit "interface" construct, but interfaces are usually implemented by abstract classes. 
- An interface is a guarantee that an object will implement a certain set of functionality.
- The abstract class that defines an interface should not implement that functionality itself.
- The functionality defined in an interface should be minimal to describe what is needed from objects of this kind. 

### **KISS: Keep It Simple, Stupid**

The principle of _KISS_ is straight-forward: try to keep your programs simple and understandable. Don't obfuscate your code with complex structures that you don't need or use, but keep things direct and make sure the structure of your code reflects the concepts that you are trying to model. 

### **DRY: Don't Repeat Yourself**

The basic principle here is to organise your code into re-usable blocks rather than rewriting the same functionality repeatedly. If you have some calculation that appears multiple times throughout your code, you can place that calculation inside a function which is called in different places to avoid repeating that logic. 

## Decoupling Code with Abstract Classes & Dependency Injection 

Dependency injection is a commonly used technique to make a pair of classes which depend on one another _loosely coupled_, i.e. to make changes to one class as independent of the other class as possible. 

Consider for example the case where we have one class which contains an instance of another.
```cpp
class Bar
{
    public:
    void print()
    {
        cout << "BAR" << endl;
    }
};

class Foo
{
    public:
    Foo()
    {
        myBar = std::unique_ptr<Bar>(new Bar());
    }

    void printBar()
    {
        myBar->print();
    }

    private:
        std::unique_ptr<Bar> myBar;
};
```
- The definition of class `Foo` is dependent on the definition of class `Bar`. 
- The constructor for `Foo` calls the constructor of `Bar` directly; if the constructor of `Bar` changes then the class `Foo` must also be changed. 
- The class `Bar` may develop and contain functionality that is irrelevant to what `Foo` needs. 

Dependency injection is generally achieved by using an abstract class in place of a concrete type for a component of a class. The abstract class defines a interface that must be met by any class that you want to use, but does not enforce what exactly that class should be. This allows you to design a class which can be reused with different components which fulfil the same functionality depending on what you need it for. 

```cpp
class AbstractBar
{
    public:
    virtual void print() = 0;
};

class Bar : public AbstractBar
{
    public:
    void print()
    {
        cout << "BAR" << endl;
    }
};
```
- `AbstractBar` is an abstract class, because its function `print` is not implemented. 
- `print` is _pure_ and _virtual_ which means that it will always be overridden by a derived class. This defines a "contract": a set of functionality that anything which inherits from this abstract class _must_ implement. We can use such abstract classes to define minimal functionality required by other classes: this is sometimes referred to as an "interface". 
    - Interfaces are a core language feature of some other languages like Java and C#, but are not explicitly implemented in C++. 
    - In C++ we generally implement interfaces using abstract classes containing only pure virtual functions and variables. 

The trick with dependency injection is to the then pass (or "inject") the component you want to use to a constructor or setter function. This is done at runtime rather than compile time, and means that different instances of the class can be instantiated with different components based on run-time considerations. 
```cpp
class Foo
{
    Foo(unique_ptr<AbstractBar> &inBar)
    {
        myBar = std::move(inBar);
    }

    void printBar()
    {
        myBar->print();
    }

    private:
        std::unique_ptr<AbstractBar> myBar;
};
```

- Now `Foo` works with an abstract class `AbstractBar`, which does not itself contain an implementation of `print`. 
- Note that the `Foo` class now does not call the constructor for the `myBar` object: the `Bar` implementation can change completely as long as it still implements the `print` method, which is the only thing that we need from it in this example. 

We gain even more flexibility by using a setter function. With this kind of structure we can also create classes that allow components to be swapped out during the lifetime of the object, meaning that the functionality of the object can be changed during runtime. 
```cpp
class Foo
{
    public:
    Foo(unique_ptr<AbstractBar> &inBar)
    {
        myBar = std::move(inBar);
    }

    void setBar(unique_ptr<AbstractBar> &inBar)
    {
        myBar = std::move(inBar);
    }

    void printBar()
    {
        myBar->print();
    }

    private:
        std::unique_ptr<AbstractBar> myBar;
};
```

## Example: Strategy Pattern

One way that we can make use of this kind of class structure is to be able to select different solutions for the same problem at runtime. Often the best solution to use for a particular problem will depend on the details of that problem: some sorting algorithms are faster on shorter lists while other are faster on longer lists, or some integrators might be more accurate and efficient when integrating slowly changing functions but others work better for oscillating functions. If we have a class which needs to have a component which achieves a task like sorting or integrating, then rather than having multiple versions of that class with different concrete implementations of these algorithms built in, we can have a class which contains an abstract sorter or integrator class. We can then have different sub-classes of our abstract class that implement the different solutions to our problem, and we can pass different solutions into our main class, or even change solutions throughout the runtime based on considerations that we can't know at compile time (e.g. the length of a list that needs to be sorted). This is commonly known as the _Strategy Pattern_ (a "pattern" is just a term for a general purpose solution which is commonly applied in programming). 

## Example: Factory Pattern 

When dealing with abstract classes it is sometimes useful to be able to make objects of different sub-classes depending on runtime considerations. In this case, we can define another class or method, sometimes known as a "factory", which returns something of the base type. Let's say we have a system that allows a person to register with the University as either a `Student` or an `Employee`, both of which inherit from a generic `Person` class. Whether or not we create `Student` or `Employee` object will depend on the input that the person gives us, which we cannot know before run time. We can then create a class or function which returns a `Person` type, but which, depending on the information input, may create a `Student` or `Employee` object and return that.  

## Implementing Multiple Interfaces 

 When working with interfaces which define minimal behaviour, it is possible that useful objects will need to implement more than one set of behaviour. This is easy to do when the sets of behaviour are nested i.e. we can model it with a chain of inheritance. (An `Undergraduate` is a type of `Student` which is a type of `UniversityMember`.) Things can be slightly more complicated when an object implements two different functionalities which are independent of one another, for example a `StudentTeachingAssistant` is a `Student` and an `Employee`, but a `Student` is not a type of `Employee` (and vice versa). 
 
 In this case we would need to implement two interfaces independently, which can be done using multiple inheritance. Multiple inheritance is a complex topic in C++ that goes beyond implementing multiple abstract interfaces, so you should think carefully about whether, and how, you use it. 

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