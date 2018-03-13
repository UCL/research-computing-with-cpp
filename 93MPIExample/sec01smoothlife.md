---
title: Smooth Life
---

## Smooth Life

### Conway's Game of Life

The Game of Life is a cellular automaton devised by John Conway in 1970.

A cell, on a square grid, is either alive or dead.
On the next iteration:

* An alive cell:
    * remains alive if it has 2 or 3 alive neighbours out of 8
    * dies (of isolation or overcrowding) otherwise
* A dead cell:
    * becomes alive if it has exactly 3 alive neighbours
    * stays dead otherwise

### Conway's Game of Life

This simple set of rules produces beautiful, complex behaviours:

!["Gospers glider gun" by Kieff. Licensed under CC BY-SA 3.0 via Wikimedia Commons.]({% figurepath %}gun.{% if site.latex %}png{%else%}gif{% endif%})

### Smooth Life

Smooth Life, proposed by Stephan Rafler, extends this to continuous variables:

* The neighbourhood is an integral over a ring centered on a point, $R(r)$.
* The local state is an integral over the disk inside the ring, $D(r)$.

![A cell and its neighbourhood]({% figurepath %}cell_neighbourhood.png)

(The ring has outer radius 3*inner radius, so that the area ratio of 1:8 matches
the grid version.)

### Smooth Life

* Each point has some degree of 'aliveness' in [0, 1]
* At the next timestep
    - a 'fully alive' cell ($D(r)=1$) remains alive if the neighbourhood is within the 'death' limits,
      i.e. $d^{(1)} \leq R(r) \leq d^{(2)}$
    - a 'fully dead' cell ($D(r)=0$) becomes alive if the neighbourhood is within the 'birth' limits,
      i.e. $b^{(1)} \leq R(r) \leq b^{(2)}$
* We actually define the new aliveness $S(D(r),R(r))$ as a smoothly varying function blending between
  these limiting cases, using a 'sigmoid' shape

### Smooth Life on a computer

We discretise Smooth Life using a grid, so that the integrals become sums.
The aliveness variable becomes a floating point number.

To avoid the hard-edges of a "ring" and "disk" defined on a grid, we weight the sum
by the fraction of a cell that would fall inside the ring or disk:

If the distance $d$ from the edge of the ring is within 0.5 units,
we weight the integral by $2d-1$, so that it smoothly varies from 1 just inside to 0 just outside.

### Smooth Life

Smooth Life shows even more interesting behaviour:

[SmoothLifeVideo](https://www.youtube.com/watch?v=KJe9H6qS82I)

* Gliders moving any direction
* "Tension tubes"

### Exercise

We will create the following two functions:

- a function to compute the two integrals $D(r)$ and $R(r)$ as
$D(x0, y0) = sum_{i, j \in \mathrm{grid}} F(i, j) \mathrm{Disk}(r=||(i - x0, j - y0)||)$
with $F(i, j)$ the current value at point $(i, j)$

- the main update function:

    loop over all points $(x, y)$ in field:

    - compute integrals centered at $(x, y)$
    - update the field using the transition function

All other functions, including the transition function, $\mathrm{Disk}$, etc are given.

### Square domain into a 1-d vector

We wrap the 2-d grid into a 1-d vector in **row-major** format:

$F(i, j) <==> F(I)$

with $i = I / {N\_x}$ and $j = I \% {N\_x}$,

with $N\_x$ the number of points in direction $x$.

### Distances wrap around a torus

We use periodic boundary conditions: the field is a torus.

{% fragment Torus_Difference, cpp/serial/src/Smooth.cpp %}

### Smoothed edge of ring and disk

$\mathrm{Disk}(r)$:

{% fragment Disk_Smoothing, cpp/serial/src/Smooth.cpp %}

### Automated tests for mathematics

{% fragment Sigmoid_Signature, cpp/serial/src/Smooth.h %}

{% fragment Sigmoid_Test, cpp/serial/test/catch.cpp %}

### Comments

We can see that this is pretty slow:

If the overall grid is $M$ by $N$, and the range of interaction (3* the inner radius), is $r$, then each time
step takes $MNr^2$ calculations: if we take all of these proportional as we "fine grain" our
discretisation (a square domain, and a constant interaction distance in absolute units), the problem grows
like $N^4$!

So let's parallelize!
