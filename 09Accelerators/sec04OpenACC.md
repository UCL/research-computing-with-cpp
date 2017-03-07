---
title: Vectorization
---

## Just USES

Using Somebody Else's Software is a good way to avoid becoming a GPU expert.
But which? Check for usability and sustainability:

* Does it do what you need?
* What license is it under?
* Is it simply available on Github/BitBucker/Gitlab?
* Are there automatic tests?
* Is it an active development repo? (number of commits, date of last commmit)
* How many people/labs/companies are committing to it? (check pull-requests)
* Is there a community of users? (check issues, wiki)
* Is there documentation?

### Broadcasting (Vectorization)

GPU require running the *same operations* over mutliple data (SIMD). Broadcasting
transforms nested loops into a set of matrix or array operations amenable to
SIMD, *and* likely to be already GPU-ized by external libraries.

Also useful for MATLAB, Python/numpy, etc...

Example:

$$G = max(||A - R_i||)$$

for all $i$, where $A$ and $R_i$ are vectors

Naive solution with explicit loops and STL:

{% fragment naive, cpp/arrayfire/max_norm.cc %}

### Broadcasting (Vectorization)

1. transform A from a vector to a matrix A3xn
1. compute pow2 = (R - A3xn)^2 elementwise
1. sum pow2 over columns
1. reduce final vector using max

{% fragment broadcasting, cpp/arrayfire/max_norm.cc %}

### A word on transfer rate

- Transfer to GPU takes time $T_0$

{% fragment transfer to gpu, cpp/arrayfire/max_norm.cc %}

- Compute takes $\frac{C}{n}$, n the number of GPU threads

{% fragment compute, cpp/arrayfire/max_norm.cc %}

- Transfer to CPU takes time $T_1$

{% fragment transfer to gpu, cpp/arrayfire/max_norm.cc %}

- Then, possibly $T_0 + T_1 + C/n > C$


### Exercise: Broadcasting

`G = \sum_{i, j, k} cos(K_k \cdot (R_i - R_j))`

Given

``` cpp
std::vector<std::array<double, 3>> const Rs = {...};
std::vector<std::array<double, 3>> const Ks = {...};
```

Write code to compute G (pseudo, or real code):

1. Create an Rs array of (3, 1, n) using `af::moddims` (`n = Rs.size()`)
1. Create an Rs array of (3, n, 1) using `af::moddims`
1. Tile appropriately
1. Compute the dot product using `af::moddims`, `af::matmul`, `af::transpose`
1. sum over the cosine of the result


### Exercise: Line of Sight

Given an n by m matrix of altitudes, with n orientations and m distances:

``` cpp
std::vector<double> altitudes(nOrientations * mDistances) = { ... };
```

Compute the whether any point i, j is in sight.
A point is in sight if, for that orientation, all previous angles between
X-horizon and X-point are smaller

                /---\
    X---\      /     \---A
         \____/

Angles are computed using the formula `atan(z_i / (stepsize * i))`.

`af::scanf` might come in handy:
- given [a0, a1, a2, ..., aN]
- it computes [0, a0, a0 + a1, a0 + a1 + a2, ..., a0 + ... + aN]
- leading zero is removed in exclusive scans

This function looks intrinsically difficult to parallelize, but there are well
known solutions. It is useful in many fields.
