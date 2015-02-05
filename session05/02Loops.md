---
title: Parallelizing loops with OpenMP
---

## Parallelizing loops with OpenMP


### Simple example


Integrate: 
$\pi = \int_0^1 \frac{4}{1+x^2} \mathrm{d}x$

Don't do this in production code. Use a standard library function to integrate functions.
{{cppfrag('05','forloop/openmpforloop.cc')}}

### Details

* Important that `x` and `sum` are private. 
    - Try making them shared and see what happens.
* Note that the default is shared.
    - Can be controlled with the default clause. 
    - `default(none)` is safer.
    - "Explicit is better that implicit".
* We use a critical region to add safely without a race condition.

### Reduction

* Aggregating a result from multiple threads with a single mathematical operation.
* Is a very common pattern.
* OpenMP has build in support for doing this. 
* Simplifies the code and avoids the explicit critical region.
* Easier to write and may perform better. 

### Reduction example.

{{cppfrag('05','forloop/openmpforloopreduction.cc')}}
