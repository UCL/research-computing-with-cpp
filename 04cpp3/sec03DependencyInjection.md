---
title: Dependency Injection
---

{% idio cpp %}

## Dependency Injection

### Construction

* What could be wrong with this:

{% code snippets/constructorDependency.cc %}


### Unwanted Dependencies

* If constructor instantiates class directly:
    * Hard-coded class name
    * Duplication of initialisation code


### Dependency Injection

* Read Martin Fowler's [Inversion of Control Containers and the Dependency Injection Pattern](http://www.martinfowler.com/articles/injection.html)
* Type 2 - Constructor Injection
* Type 3 - Setter Injection


### Constructor Injection Example

{% code snippets/constructorInjection.cc %}


### Setter Injection Example

{% code snippets/setterInjection.cc %}

Question: Which is better?


### Advantages of Dependency Injection

* Using Dependency Injection
    * Removes hard coding of `new ClassName`
    * Creation is done outside class, so class only uses public API
    * Leads towards fewer assumptions in the code

### Homework 18

* Taking the `Student` class from homework 16: 
   * Create a new `Laptop` class that has a string `os` data member for the operating system name and an integer  `year` data member for the year produced. 
   * `Laptop` should have both a default constructor that sets `year` to 0 and name to "Not set" as well as an overloaded constructor that initialises both `year` and  `os`
   * Modify `Student` to have a `Laptop` data member 
   * Try out the two types of dependency injection above: constructor, setter
   * Confirm that the `Student` class is now invarient to changes in how you instantiate `Laptop`    

{% endidio %}
