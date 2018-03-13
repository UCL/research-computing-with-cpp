---
title: Derived Datatypes
---

## Derived Datatypes

### Avoiding buffering

We've still got our ugly re-buffering of the halo data into contiguous memory,
arising from our use of `vector<vector< double > >`

We can get around that by changing to a flat 1-d array of memory, and storing the
$(x,y)$ element at `Field[`$Mx+y$`]`. This works fine, especially if we refactor
all access to get and update from the field into accessors, so we only need make
the change in one place.

### Wrap Access to the Field

{% idio cpp/parallel/src/Smooth.cpp %}

{% fragment Wrap_Access %}

### Copy Directly without Buffers

{% fragment Unbuffered_Send %}

### Defining a Halo Datatype

It's usually the case in programming that thinking of a whole body of data as a single
entity produces cleaner, faster code than programming at the level of the individual datum.

We want to be able to think of the **Halo** as a single object to be transferred, rather
than as a series of `double`s.

We can do this using an MPI Derived Datatype:

### Declare Datatype

{% fragment Define_Datatype %}

### Use Datatype

{% fragment Use_Datatype %}

{% endidio %}

### Strided datatypes

Supposing we wanted to use a 2-d decomposition. Our y-direction halo's data would not be contiguous
in memory.

Let's imagine we used $Ny+x$ to index into the field instead of $Mx+y$: we can define a derived datatype
which specifies data as a series of stretches, with gaps.

`MPI_Type_Vector(`$M,r,N$`)` would define the relevant type for this: $M$ chunks, each $r$ `double`s
long, each $N$ `double`s apart in memory.
