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

!["Gospers glider gun" by Kieff. Licensed under CC BY-SA 3.0 via Wikimedia Commons.]({% figurepath %}gun.{% if site.config.latex %}png{%else%}gif{% endif%})

### Smooth Life

Smooth Life, proposed by Stephan Rafler, extends this to continuous variables:

* The neighbourhood is an integral over a ring centered on a point.
* The local state is an integral over the disk inside the ring.

(The ring has outer radius 3*inner radius, so that the area ratio of 1:8 matches
the grid version.)

### Smooth Life

* A point has some degree of aliveness.
* Next timestep, a point's aliveness depends on these two integrals. (`$F_{r}$` and `$F_{d}$`)
* The new aliveness `$S(F_{r},F{d})$` is a smoothly varying function such that:
    * If `$F_{d}$` is 1, S will be 1 if `$d_1< F_{r} < d_2$`
    * If `$F_{d}$` is 0, S will be 1 if `$b_1< F_d < b_2$`

A "Sigmoid" function is constructed that smoothly blends between these limits.

### Smooth Life on a computer

We discretise Smooth Life using a grid, so that the integrals become sums.
The aliveness variable becomes a floating point number.

To avoid the hard-edges of a "ring" and "disk" defined on a grid, we weight the sum
by the fraction of a cell that would fall inside the ring or disk:

If the distance $d$ from the edge of the ring is within 0.5 units,
we weight the integral by $2d-1$, so that it smoothly various from 1 just inside to 0 just outside.

### Smooth Life

Smooth Life shows even more interesting behaviour:

[SmoothLifeVideo](https://www.youtube.com/watch?v=KJe9H6qS82I)

* Gliders moving any direction
* "Tension tubes"

### Serial Smooth Life

Have a look at our [serial](https://github.com/UCL/SmoothLifeExample) implementation of SmoothLife.

We can see that this is pretty slow:

If the overall grid is $M$ by $N$, and the range of interaction (3* the inner radius), is $r$, then each time
step takes $MNr^2$ calculations: if we take all of these proportional as we "fine grain" our
discretisation (a square domain, and a constant interaction distance in absolute units), the problem grows
like $N^4$!

To make this faster, we'll need to parallelise with MPI. But let's look at a few interesting things
about the serial implementation.

### Main loop

Four levels deep:

{% idio cpp/serial/src/Smooth.cpp %}

{% fragment Main_Loop %}

### Swapped before/after fields

{% fragment Swap_Fields %}



### Distances wrap around a torus

{% fragment Torus_Difference %}

### Smoothed edge of ring and disk.

{% fragment Disk_Smoothing %}

{% endidio %}

### Automated tests for mathematics

{% fragment Sigmoid_Test, cpp/serial/test/catch.cpp %}
