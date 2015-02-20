---
title: Halo Swap
---

## Halo Swap

### Where do we put the communicated data?

The data we need to receive from our neighbour needs to be put in a
place where it can be conveniently used to calculate the new state
of cells within distance $r$ of the boundary.

The standard design pattern for this is to use a **Halo Swap**:
we extend the memory buffer used for our state, adding new space either
side of our main domain to hold a **halo**.

### Domains with a halo

Thus, each process now holds $N+2r$ cells:

* $0 \le x < r$, the left halo, holding data calculated by the left neighbour
* $r \le x < 2r$, data which we calculate, which will form our **left neighbour's right halo**
* $2r \le x < N$, data which we calculate, unneeded by neighbours
* $N \le x < N+r$, data which we calculate, which will form our **right neighbour's left halo**
* $N+r \le x < N+2r$, the right halo.

### Coding with a halo

We will thus **update** the field only from $r$ to $N+r$, but we will **access** the field
from $0$ to $N+2r$:

// Fragment here.

### Transferring the halo

We need to pass data left at the same time as we receive data from the right.

If we use separate `Send` and `Recv` calls, we'll have a deadlock: process 2 will be sending
to process 1, while process 3 is sending to process 2. No process will be executing a `Recv`.

Fortunately, MPI provides `Sendrecv`: expressing that we want to do a blocking `Send` to
one process while we simultaneously do a blocking `Recv` from another. Exactly what we need
for pass-the-parcel.

// Fragment here

### Noncontiguous memory

Our serial solution uses a C++ `vector<vector<double> >` to store our field.

That means each column of data starts at a different location in memory; our data is not
contiguous in memory, the outer vector holds a series of pointers to the start of each
inner vector.

This means we can't just transmit $Mr$ `doubles` in one go by sending from the address of
`&field1[0][0]`.

### Buffering

We'll get around this by copying all the data into a send buffer, and unpacking from a
receive buffer. This adds to our communication overhead, but doesn't change its scaling behaviour,
both overheads are $O(Mr)$.

We'll look in a later section how this can be avoided.

### Buffering

// Fragment here


