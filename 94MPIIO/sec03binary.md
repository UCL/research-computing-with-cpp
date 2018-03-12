---
title: Binary IO
---

## Binary IO in C++

### Binary IO in C++

To do basic binary file IO in C++, we set `std::ios::binary` as we open the file.

However, we can't use the nice `<<` operator, as these still generate formatted `char`s.
We have to use `ostream::write()`, which is a low-level, C-style function.

### Binary IO in C++

{% idio ../07MPIExample/cpp/parallel %}

{% fragment Header, src/BinaryWriter.cpp %}

{% fragment write, src/BinaryWriter.cpp %}


### Binary IO in NumPy

{% fragment BinaryRead, visualise.py %}

{% endidio %}

... more on that 'bulk_type' parameter later.
