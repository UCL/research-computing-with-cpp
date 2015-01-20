---
title: Class Templates
---

## Class Templates

### Example

* If you understand template functions, then template classes are easy!
* Refering to [this tutorial][TemplateClassTutorial], an example: 
* Header:
{{cppfrag('02','pairClassExample/pairClassExample.h')}}
* Implementation:
{{cppfrag('02','pairClassExample/pairClassExample.cc')}}
* Usage:
{{cppfrag('02','pairClassExample/pairClassMain.cc')}}


### Quick Comments

* Implementation, 3 uses of parameter T
* Same Implicit/Explicit instantiation rules
* Note implicit requirements, eg. operator >

### Template Specialisation

* If template defined for type T
* Full specialisation - special case for a specific type eg. char
* Partial specialisation - special case for a type that still templates, e.g. T* 

```
template <typename T> class MyVector {
template <> class MyVector<char> {  // full specialisation
template <typename T> MyVector<T*> { // partial specialisation
```

[TemplateClassTutorial]: http://www.cplusplus.com/doc/tutorial/templates/ 'Template Class Tutorial'
