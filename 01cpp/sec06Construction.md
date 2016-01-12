---
title: Construction
---

{% idio cpp %}

## Construction

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


### Constructional Patterns

* Other methods include
    * [Builder Pattern](https://en.wikipedia.org/wiki/Builder_pattern)
    * [Factory Pattern](https://en.wikipedia.org/wiki/Factory_method_pattern)
    * [Abstract Factory Pattern](https://en.wikipedia.org/wiki/Abstract_factory_pattern)
    * See [Gang of Four](https://en.wikipedia.org/wiki/Design_Patterns) book
* Also look up [Service Locator Pattern](https://en.wikipedia.org/wiki/Service_locator_pattern)
* These all work well with dependency injection


{% endidio %}
