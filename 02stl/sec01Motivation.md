---
title: Motivation
---

{% idio cpp %}

## Motivation


### Motivation - why STL?

Because you have better things to do than rediscovering the wheel.

Task: Read in two files with an unknown number of integers, and sort them altogether.


### Brute-force solution

{% code sortArrays.cc %}

Output:
{% code sortArrays.out %}


### STL solution

{% code sortArraysCont.cc %}

Output:
{% code sortArraysCont.out %}


### More conveniences

* Similar API between different STL containers.
  eg. myContainer->begin(), end(), at(), erase(), clear(), size().....
* Algorithms and data structures optimised for speed and memory.

{% endidio %}
