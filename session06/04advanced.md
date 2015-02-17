---
Title: More advanced MPI
---

## More advanced MPI

### Architecture and usage

* depending on library, MPI processes can be *placed* on specific node...
* ... and even chained to specific cores
* Fewer processes than core means we can do MPI + openmp:
   - some data is distributed (MPI)
   - some data is shared (openMP)
* MPI-3 allows for creating/destroying processes dynamically

### Splitting Communicators

``MPI_Group_*`` specify operations to create sets of processes.
In practice, it defines operations on sets:

- union
- intersection
- difference

And allows the creation of a communicator for the resulting group.

### More MPI data-types

It is possible to declare complex data types

- strided vectors, e.g. only one of every element (MPI_Type_vector)
- sub-matrices (strided in n axes, n >= 2) (MPI_Type_Create_struct)
- irregular strides (MPI_Type_indexed)


### One sided communication

Prior to MPI-3, both sending and receiving processes must be aware of the
communication.

One-sided communication allow processes to define a buffer that other processes
can access without their explicit knowledge.


### And also

- Cartesian grid topology where process (1, 1) is neighbor
  of (0, 1), (1, 0), (2, 1), (1, 2). With simplified operations to send data
  EAST, WEST, UP, DOWN....
- More complex graph topologies
- non-blocking collective operations
