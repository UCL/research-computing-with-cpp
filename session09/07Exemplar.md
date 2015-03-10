---
title: Another CUDA Example
---

##Another CUDA Example

###Monte Carlo PI

* There's a very slow way to calculate $\pi$ using a fair random number generator.
* Consider throwing darts at a square wall with a dartboard in the middle.
* Your darts are evenly distributed inside the square of side $2r$
* The chance of a dart hitting the dartboard is $\pi r^2/4r^2$ = $\pi/4$

### CUDA with MPI

* The [Emerald](http://www.cfi.ses.ac.uk/emerald/) supercomputer has a GPU on every node.
* So we can use MPI with CUDA
* The world's fastest computers use accelerators in conjuction with MPI

###Let's implement this using CUDA and thrust.

* [See GitHub](https://github.com/UCL-RITS/emerald_play/blob/master/thrust_monte_pi/monte_pi.cu)
