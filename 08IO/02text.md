---
title: Formatted Text IO
---

## Formatted Text IO

### Formatted Text IO

* Is easy
* Is portable
* Is human-readable
* Is slow
* Wastes space

### When and when not to use text IO

Use text IO:

* Metadata
* Configuration
* Logging

Don't use it:

* For output of large numerical datasets
* For initial conditions
* For checkpoint/restart

### Use libraries to generate formatted text

For a templating library like Mako in C++ try [CTemplate](https://code.google.com/p/ctemplate/)

This is a great way to create XML and YAML files.

Raw CSV file generation with built-in C++ `<iostream>` is not very robust.

Libraries will automatically quote strings which contain commas.

Boost's [Spirit](http://boost-spirit.com/home/) library is good for this too.

### Low-level IO exemplar

Nevertheless, in simple cases where text is of a known format, C++'s built in formatted IO
is quick and easy:

{% idio ../07MPIExample/cpp/parallel/src/TextWriter.cpp %}

{% fragment Header %}
{% fragment Write %}

{% endidio %}

### Text Data and NumPy

Parsing text data in NumPy is also very easy:

{% idio ../07MPIExample/cpp/parallel/visualise.py %}

{% fragment TextRead %}
{% fragment Reshape %}

{% endidio %}

this is how the animations you saw last lecture were created.

### Text File Bloat

However, representing $1.3455188104 \cdot 10^{-10}$ in text:

> `1.3455188104e-10, `

uses many bytes per number, (one per character in ASCII, more in unicode)
whereas recording in a binary representation, even at double precision, typically uses 8.

It's also harder do parallel IO in such files, as the distance of a certain quantity from the
start of the file can't be predicted.
