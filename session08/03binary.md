---
title: Binary IO
---

## Binary IO in C++

### Binary IO in C++

To do basic binary file IO in C++, we set `std::ios::binary` as we open the file.

However, we can't use the nice `<<` operator, as these still generate formatted `char`s.
We have to use `ostream::write()`, which is a low-level, C-style function.

### Binary IO in C++

{{cppfrag('07','parallel/src/BinaryWriter.cpp','Header')}}
{{cppfrag('07','parallel/src/BinaryWriter.cpp','Write')}}


### Binary IO in NumPy

```python
{{d['session07/cpp/parallel/visualise.py|idio|t']['BinaryRead']}}
```

... more on that 'bulk_type' parameter later.
