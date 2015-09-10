---
title: Portability
---

## Endianness and Portability

### Portability

There's a big problem with binary files: the bytes that get written for a `double`
on one platform are different from those on others.

This is particularly problematic in HPC: the architecture of your laptop is unlikely
to be the same as on ARCHER, so if when ship your output files back, your nice visualisation
code will not work.

### Length of datatypes

The C++ standard **does not** specify how many bytes are used to represent a `double`: just that
it is more than a `float`! (And similarly for other datatypes.)

This means that your data will be represented differently on different
platforms.

Therefore, don't forget to use `sizeof(type)` whenever working with byte-level routines
like MPI, if you want your code to work on both your laptop, and your local supercomputer.

### Endianness

Another problem is Endianness: a `double` is actually **almost** always 8 bytes, with
a 1 bit sign, 11 bit exponent, and 52 bit mantissa. (The IEEE standard).

But there's still ambiguity in how these bytes are ordered.

The 4-byte standard signed integer "5" can be represented as:

> 00000101 00000000 00000000 00000000 (little endian)

or as

> 00000000 00000000 00000000 00000101 (big endian)

### NumPy dtypes

As long as you're aware of these problems, you can usually make your visualiser compatible with
the endianness and byte counts of the data you're getting off your system.

If you're visualising with python you can set your datatype to be e.g. `<f8` for a little endian
8-byte floating point, or `>i4` for a big-endian 4-byte signed integer.

### XDR

XDR, or 'extensible data representation' is a portability standard defined for binary IO. If you
write through XDR, data will be converted to the XDR standard representation.

XDR data is big endian, has everything in multiples of 4 bytes, and supports a rich library of
appropriate types.

### Writing with XDR

{% idio ../session07/cpp/parallel/src %}

{% fragment Includes, XdrWriter.h %}
{% fragment Write, XdrWriter.cpp %}
{% fragment Create, XdrWriter.cpp %}

{% endidio %}
