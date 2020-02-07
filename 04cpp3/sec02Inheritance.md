---
title: Inheritance
---

{% idio cpp %}

## Inheritance

### Don't overuse Inheritance

* Inheritance is not just for saving duplication
* It MUST represent derived/related types
* Derived class must truely represent 'is-a' relationship
* eg 'Square' is-a 'Shape'
* Deep inheritance hierarchies are almost always wrong
* If something 'doesn't quite fit' check your inheritance


### Surely Its Simple?

* Common example: Square/Rectangle problem, [here](http://www.oodesign.com/liskov-s-substitution-principle.html)

{% code construction/square.cc %}


### Liskov Substitution Principal

* [Wikipedia](https://en.wikipedia.org/wiki/Liskov_substitution_principle)
* "if S is a subtype of T, then objects of type T may be replaced with objects of type S without altering any of the desirable properties of that program"
* i.e. is S is derived class from base class T then objects of type T should be able to be replaced with objects of type S 
* Can something truely be substituted?
* If someone else filled a vector of type T, would I care what type I have?

### What to Look For

* If you have:
    * Methods you don't want to implement in derived class
    * Methods that don't make sense in derived class
    * Methods that are unneeded in derived class
    * If you have a list of something, and someone else swapping in a derived type would cause problems
* Then you have probably got your inheritance wrong


### Composition Vs Inheritance

* Lots of Info online eg. [wikipedia](https://en.wikipedia.org/wiki/Composition_over_inheritance)
* In basic OO Principals
    * 'Has-a' means 'pointer or reference to'
    * eg.`Car` has-a `Engine`
* But there is also:
   * [Composition](https://en.wikipedia.org/wiki/Object_composition#Composition): Strong 'has-a'. Component parts are owned by thing pointing to them.
   * [Aggregation](https://en.wikipedia.org/wiki/Object_composition#Aggregation): Weak 'has-a'. Component part has its own lifecycle.
   * Association: General term, referring to either composition or aggregation, just a 'pointer-to'

 
### Examples

* House 'has-a' set of Rooms. Destroying House, you destroy all Room objects. No point having a House on its own.
* Room 'has-a' Computer. If room is being refurbished, Computer should not be thrown away. Can go to another room


### But Why?

* Good article: [Choosing Composition or Inheritance](https://www.thoughtworks.com/insights/blog/composition-vs-inheritance-how-choose)
* Composition is more flexible 
* Inheritance has much tighter definition than you realise

{% endidio %}
