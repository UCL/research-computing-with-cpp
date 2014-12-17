---
title: Class Templates
---

## Class Templates

If you understand template functions, then template classes are easy!

### Example

Refering to [this tutorial][TemplateClassTutorial], an example: 

Header:
{{cppfrag('02','pairClassExample/pairClassExample.h')}}

Implementation:
{{cppfrag('02','pairClassExample/pairClassExample.cc')}}

Usage:
{{cppfrag('02','pairClassExample/pairClassMain.cc')}}

### Quick Comments

* Implementation, 3 uses of parameter T
* Same Implicit/Explicit instantiation rules
* Note implicit requirements, eg. operator >

[TemplateClassTutorial]: http://www.cplusplus.com/doc/tutorial/templates/ 'Template Class Tutorial'