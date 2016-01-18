---
title: Object-Oriented Tips
---

{% idio cpp %}

## Object-Oriented Tips

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
* Can something truely be substituted?
* Look for:
  * Preconditions cannot be strengthened
  * Postconditions cannot be weakened
  * Invariants preserved
  * History constraint (e.g. mutable point deriving from immutable point)


### What to Look For

* If you have:
  * Methods you don't want to implement in derived class
  * Methods that don't make sense in derived class
  * Methods that are unneeded in derived class
  * If you have a list of something, and someone else swapping a derived type would cause problems
* Then you have probably got your inheritance wrong


 
{% endidio %}
