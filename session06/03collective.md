---
title: Collective communication
---

## Collective Communication

### Many possible communication schemes

Think of two possible forms of *collective* communications:

- give a beginning state
- give an end state

![](session06/figures/collective)

### Broadcast: one to many

--------------------------- ---------------------------------
data in 0, no data in 1, 2  ![](session06/figures/broadcast0)
data from 0 sent to 0, 1    ![](session06/figures/broadcast1)
--------------------------- ---------------------------------

### Gather: many to one

------------------------- ---------------------------------
data in 0, 1, 2           ![](session06/figures/collective)
data from 1, 2 sent to 0  ![](session06/figures/gather1)
------------------------- ---------------------------------

### Scatter: one to many

------------------------- ---------------------------------
data in 0                 ![](session06/figures/gather1)
data from 0 in 0, 1, 2    ![](session06/figures/collective)
------------------------- ---------------------------------

### All to All: many to many

------------------  -------------------------------
data in 0, 1, 2     ![](session06/figures/all2all0)
from each to each   ![](session06/figures/all2all1)
------------------  -------------------------------


### Reduce operation

----------------- ---------------------------------
data in 0, 1, 2   ![](session06/figures/collective)
Baby Bunny!       ![](session06/figures/reduce1)
----------------- ---------------------------------

Wherefrom the baby bunny?

. . .

Sum, difference, or any other *binary* operator:

![](session06/figures/BunnyOps)

### Collective operation API

Group synchronisation:

``` cpp
int MPI_Barrier(MPI_Comm comm);
```

Broadcasting:

``` cpp
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
    MPI_Comm comm)
```

----------    -----------------------------------------------------------------------
buf           Pointer to sending/receiving buffer

count         Size of the buffer/message

datatype      Informs on the type of the buffer

root          Sending processor

comm          The communicator!

return        Error tag
----------    -----------------------------------------------------------------------

### Example of collective operation (1)

Insert into a new CATCH section the following commands

{{cppfrag("06", "collective.cc", segment="broadcast")}}

### Example of collective operations (2)

And then insert the following right after it

{{cppfrag("06", "collective.cc", segment="tests")}}

### Causing deadlocks

Explain why the following two codes fail.

1. Replace the loop in the last fragment with:

``` Cpp
for(int i(1); i < size; ++i) ...
```

2. Refactor and put everything inside the loop

``` Cpp
std::string const peace = "I come in peace!";
std::string message;
for(int i(0); i < size; ++i) {
    if(i == 0 and rank == 0) { /* broadcast */ }
    else if(rank == i) { /* broadcast */ }
    if(rank == i) { /* testing bit */ }
    MPI_Barrier(MPI_COMM_WORLD);
}
```

NOTE: a loop with a condition for i == 0 is a common anti-pattern (eg bad)


### All to all operation

``` cpp
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
               MPI_Comm comm)
```

----------    -----------------------------------------------------------------------
sendbuf       Pointer to sending buffer (only significant at root)

sendcount     Size of a *single* message

datatype      Informs on the type of the buffer

recvbuf       Pointer to receiving buffers (also at root)

recvcount     Size of the receiving buffer

recvtype      Informs on the type of the receiving buffer
----------    -----------------------------------------------------------------------

Exercise:
    Have the root scatter "This....." "message.." "is.split." to 3 processors
    (including it self).


### Splitting the communicators

Groups of processes can be split according to *color*:

![](session06/figures/split)

``` cpp
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
```

----------    -----------------------------------------------------------------------
comm          Communicator that contains all the processes to be split
color         All processes with same color end up in same group
int key       Controls rank in final group
newcomm       Output communicator
----------    -----------------------------------------------------------------------

### Splitting communicators: example

The following splits processes into two groups with ratio 1:2.

{{cppfrag("06", "split.cc", "main")}}


### Splitting communicators: Exercise

Exercise:

- use "-rank" as the key: what happens?
- split into three groups with ratios 1:1:2
- use one of the collective operation on a single group


### Scatter operation solution

{{cppfrag("06", "scatter.cc", segment="scatter")}}

