---
title: Class Templates
---

## Class Templates

### Class Templates Example - part 1

* If you understand template functions, then template classes are easy!
* Refering to [this tutorial][TemplateClassTutorial], an example:

{% idio cpp/pairClassExample %}

Header:

{% code pairClassExample.h %}


### Class Templates Example - part 2

Implementation:

{% code pairClassExample.cc %}


### Class Templates Example - part 3

Usage:

{% code pairClassMain.cc %}

{% endidio %}

### Quick Comments

* Implementation, 3 uses of parameter T
* Same Implicit/Explicit instantiation rules
* Note implicit requirements, eg. operator >
    * Remember the 2 stage compilation
    * Remember code not instantiated until its used
    * Take Unit Testing Seriously!

### Template Specialisation

* If template defined for type T
* Full specialisation - special case for a specific type eg. char
* Partial specialisation - special case for a type that still templates, e.g. T*

```
template <typename T> class MyVector {
template <> class MyVector<char> {  // full specialisation
template <typename T> MyVector<T*> { // partial specialisation
```

### Homework 19

* Implement the above class `MyPair` template
* Try out with both Implicit and Explicit instantiation
* Add a `Swap()` method that switches the contents of `m_Values[0]` and `m_Values[1]`

[TemplateClassTutorial]: http://www.cplusplus.com/doc/tutorial/templates/ 'Template Class Tutorial'
