---
title: Program To Interfaces
---

{% idio cpp %}

## Program To Interfaces

### Why?

* In research code - "just start hacking"
* You tend to mix interface and implementation
* Results in client of a class having implicit dependency on the implementation
* So, define a pure virtual class, not for inheritance, but for clean API 

### Example

{% code snippets/pureVirtual.cc %}


### Comments

* Useful between sub-components of a system
    * GUI front end, Web back end
    * Logic and Database
* Is useful in general to force loose connections between areas of code
    * e.g. different libraries that have different dependencies
    * e.g. camera calibration depends on OpenCV
    * define an interface that just exports standard types
    * stops the spread of dependencies

{% endidio %}
