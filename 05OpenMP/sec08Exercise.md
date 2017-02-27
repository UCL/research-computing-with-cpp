---
title: Exercise
---

## Wavelet decomposition

### Wavelet transforms

- [Walevet transforms][WaveletTransform] decompose a signal into details of
    increasingly large scales

- Given a signal $s[j]$, with $j \in [1, N]$, we want to split
     * an approximation $A_s[i]$ with $i \in [1, N/2]$
     * the details $D_s[i]$ with $i \in [1, N/2]$

- Repeat the process on the approximaton

### Pseudo-code


* loop over levels in $[0, L^\textrm{max}]$

For simplicity, the signal is periodic: $s[j] = s[j + N]$.

- set $N' = N$ if $N$ is even, $N'=N+1$ otherwise
- with $j \in [0, N' - 1], i \in [0, n - 1]$
- set $D^s[j] = \sum_{i=0}^{n - 1} s[2*j + i] h[i]$
- set $A^s[j] = \sum_{i=0}^{n - 1} s[2*j + i] l[i]$
- set $s = A^s$

### Instructions

Given a header and the unit-tests:

1. reconstruct the serial version using the jigsaw implementation file
2. Parallelize with openmp

How many parallelization schemes did you come up with?

If you are not having enough fun, figure out the inverse operation. Wavelets
transforms are bijective.

### Header

{% code cpp/wavelets/wavelets.h %}

### Tests

{% code cpp/wavelets/main.cc %}

### Implementation

{% code cpp/wavelets/wavelets.cc %}


[WaveletTransform]: http://en.wikipedia.org/wiki/Wavelet_transform
