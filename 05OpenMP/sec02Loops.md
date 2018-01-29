---
title: Parallelizing loops with OpenMP
---

## Parallelizing loops with OpenMP


### Simple example


Integrate:

$\int_0^1 \frac{4}{1+x^2} \mathrm{d}x=\pi$

{% code cpp/forloop/openmpforloop.cc %}


### Variable scope

* Private: Each thread has its own copy
* Shared: Only one shared variable
* Firstprivate: Private variable but initialized with serial value

### Details of example

* Important that `x` and `sum` are private
    - Try making them shared and see what happens
* Note that the default is shared
    - Can be controlled with the default clause
    - `default(none)` is safer
    - "Explicit is better that implicit"
* We use a critical region to add safety without a race condition

### Reduction

* Aggregating a result from multiple threads with a single mathematical operation
* Is a very common pattern
* OpenMP has build in support for doing this
* Simplifies the code and avoids the explicit critical region
* Easier to write and may perform better    

### Reduction example

{% code cpp/forloop/openmpforloopreduction.cc %}
