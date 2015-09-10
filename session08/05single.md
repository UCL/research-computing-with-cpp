---
title: Writing from a single process
---

## Writing from a single process

### One process, one file

So far, we've been writing one process from each file, calling them
`frames.dat.0`, `frames.dat.1` etc.

{% idio ../session07/cpp/parallel %}

{% fragment AddName, src/TextWriter.cpp %}


We've then been reconciling them together in our visualiser code:

{% fragment Suffix, visualise.py %}

{% fragment EachFile, visualise.py %}

{% fragment Concatenate, visualise.py %}

### One process, one file

This is necessary, because we **cannot** simply have multiple processes writing to the same file,
without considerable care, as they will:

* Block while waiting for access to the file
* Overwrite each others' content

However, the one-process-one-file approach **does not scale** to large numbers of files: the file system can
be overwhelmed once we're running at the thousands-of-processors level.

### Writing from a single process

An alternative approach is to share all the data to a master process. (Which can be done in $O(ln p)$ time),
and then write from that file.

This avoids the complexity of reconciling datafiles locally, and avoids overwhealming the cluster filesystem
with many small files.

### Writing from a single process

{% fragment Write, src/SingleWriter.cpp %}

### Writing from a single process

It's important to create the file and write the header only on the master process:

{% fragment Create, src/SingleWriter.cpp %}

{% endidio %}
