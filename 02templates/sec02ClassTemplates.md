---
title: Class Templates
---

## Class Templates

### Class Templates Example - 1

* If you understand template functions, then template classes are easy!
* Refering to [this tutorial][TemplateClassTutorial], an example:

{% idio cpp/pairClassExample %}

Header:

{% code pairClassExample.h %}


### Class Templates Example - 2

Implementation:

{% code pairClassExample.cc %}


### Class Templates Example - 3

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

### Nested Types

In libraries such as [ITK][ITK], we see:

```
    template< typename T, unsigned int NVectorDimension = 3 >
    class Vector:public FixedArray< T, NVectorDimension >
    {
      public:
        // various stuff
        typedef T  ValueType;
        // various stuff
        T someMemberVariable;
```

* typedef is just an alias
* using nested typedef, must be qualified by class name
* can also refer to a real variable

[TemplateClassTutorial]: http://www.cplusplus.com/doc/tutorial/templates/ 'Template Class Tutorial'
[ITK]: http://www.itk.org
