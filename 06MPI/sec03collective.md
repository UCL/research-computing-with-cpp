---
title: Collective communication
---

## Collective Communication

### Many possible communication schemes

Think of two possible forms of *collective* communications:

- give a beginning state
- give an end state

![]({% figurepath %}collective.png)

### Broadcast: one to many

| State                      | Figure                              |
|:---------------------------|:------------------------------------|
| data in 0, no data in 1, 2 | ![]({% figurepath %}broadcast0.png) |
| data from 0 sent to 0, 1   | ![]({% figurepath %}broadcast1.png) |

### Gather: many to one

| State                    | Figure                              |
|:-------------------------|:------------------------------------|
| data in 0, 1, 2          | ![]({% figurepath %}collective.png) |
| data from 1, 2 sent to 0 | ![]({% figurepath %}gather1.png)    |


### Scatter: one to many

| State                  | Figure                              |
|:-----------------------|:------------------------------------|
| data in 0              | ![]({% figurepath %}gather1.png)    |
| data from 0 in 0, 1, 2 | ![]({% figurepath %}collective.png) |

### All to All: many to many

| State             | Figure                            |
|:------------------|:----------------------------------|
| data in 0, 1, 2   | ![]({% figurepath %}all2all0.png) |
| from each to each | ![]({% figurepath %}all2all1.png) |

### Reduce operation
| State           | Figure                              |
|:----------------|:------------------------------------|
| data in 0, 1, 2 | ![]({% figurepath %}collective.png) |
| Baby Bunny!     | ![]({% figurepath %}reduce1.png)    |

Wherefrom the baby bunny?

. . .

Sum, difference, or any other *binary* operator:

![]({% figurepath %}BunnyOps.png)

### Collective operation API (1)

Group synchronisation:

``` cpp
int MPI_Barrier(MPI_Comm comm);
```

Broadcasting:

``` cpp
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
    MPI_Comm comm)
```

| Parameter | Content                             |
|:----------|:------------------------------------|
| buf       | Pointer to sending/receiving buffer |
| count     | Size of the buffer/message          |
| datatype  | Informs on the type of the buffer   |
| root      | Sending processor                   |
| comm      | The communicator!                   |
| return    | Error tag                           |


### Example of collective operation (1)

Insert into a new CATCH section the following commands

{% fragment broadcast, cpp/collective.cc %}

### Example of collective operations (2)

And then insert the following right after it

{% fragment tests, cpp/collective.cc %}

### Causing deadlocks

Explain why the following two codes fail.

1. Replace the loop in the last fragment with:

``` cpp
for (int i(1); i < size; ++i) ...
```

2. Refactor and put everything inside the loop

``` cpp
std::string const peace = "I come in peace!";
std::string message;
for (int i(0); i < size; ++i) {
    if (i == 0 && rank == 0) { /* broadcast */ }
    else if (rank == i) { /* broadcast */ }
    if (rank == i) { /* testing bit */ }
    MPI_Barrier(MPI_COMM_WORLD);
}
```

NOTE: a loop with a condition for i == 0 is a common anti-pattern (i.e. bad)


### All to all operation

``` cpp
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
               MPI_Comm comm)
```

| Parameter | Content                                              |
|:----------|:-----------------------------------------------------|
| sendbuf   | Pointer to sending buffer (only significant at root) |
| sendcount | Size of a *single* message                           |
| datatype  | Type of the buffer                                   |
| recvbuf   | Pointer to receiving buffers (also at root)          |
| recvcount | Size of the receiving buffer                         |
| recvtype  | Informs on the type of the receiving buffer          |

Exercise:
    Have the root scatter "This....." "message.." "is.split." to 3 processors
    (including itself).


### Splitting the communicators

Groups of processes can be split according to *color*:

![]({% figurepath %}split.png)

``` cpp
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
```

| Parameter | Content                                                  |
|:----------|:---------------------------------------------------------|
| comm      | Communicator that contains all the processes to be split |
| color     | All processes with same color end up in same group       |
| int key   | Controls rank in final group                             |
| newcomm   | Output communicator                                      |

### Splitting communicators: example

The following splits processes into two groups with ratio 1:2.

{% fragment main, cpp/split.cc %}


### Splitting communicators: Exercises

1. Use "-rank" as the key: what happens?
2. Split into three groups with ratios 1:1:2
3. Use one of the collective operations on a single group


### Scatter operation solution

{% fragment scatter, cpp/scatter.cc %}
