---
title: Point to Point
---

## Point to point communication

### Many point-2-point communication schemes

Can you think of two behaviours for message passing?

![]({% figurepath %}mpi.png)

- Process 0 can (i) gives message, (ii) leave, and/or (iii) wait for
  acknowledgements
- Process 1 can (i) receives message
- MPI can (i) receive message, (ii) deliver message, (iii) deliver
  acknowledgments

### Blocking synchronous send

------------------------------   ----------------------------
a. 0, 1, and MPI stand ready:    ![]({% figurepath %}sync0.png)
b. message dropped off by 0:     ![]({% figurepath %}sync1.png)
c. transit:                      ![]({% figurepath %}syncT.png)
d. message received by 1         ![]({% figurepath %}syncA.png)
e. receipt received by 0         ![]({% figurepath %}syncR.png)
------------------------------   ----------------------------

### Blocking send

------------------------------  -----------------------------
a. 0, 1, and MPI stand ready:   ![]({% figurepath %}sync0.png)
b. message dropped off by 0:    ![]({% figurepath %}sync1.png)
c. transit, 0 leaves            ![]({% figurepath %}ssyncT.png)
d. message received by 1        ![]({% figurepath %}ssyncA.png)
------------------------------  -----------------------------

### Non-blocking send

-------------------------------  -----------------------------
a. 0, 1, and MPI stand ready:    ![]({% figurepath %}async0.png)
b. 0 leaves message in safebox   ![]({% figurepath %}async1.png)
c. transit                       ![]({% figurepath %}asyncT.png)
d. message received by 1         ![]({% figurepath %}asyncA.png)
e. receipt placed in safebox     ![]({% figurepath %}asyncR.png)
-------------------------------  -----------------------------


### Blocking synchronous send


``` cpp
int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm)
```

----------    -----------------------------------------------------------------------
buf           Pointer to buffer. Always `void` because practical C is not type safe.

count         Size of the buffer. I.e. length of the message to send.

datatype      Informs on the type of the buffer. ``MPI_INT`` for integers,
              ``MPI_CHAR`` for characters.  Lots of others.

dest          Rank of the *receiving* process

tag           A tag for message book-keeping

comm          The communicator!

return        An error tag. Equals ``MPI_SUCCESS`` on success.
----------    -----------------------------------------------------------------------



### Blocking receive

``` cpp
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)
```

Good for both synchrone and asynchone communication

----------    -----------------------------------------------------------------------
buf           Pointer to receiving *pre-allocated* buffer

count         Size of the buffer. I.e. maximum length of the message to
              receive. See ``MPI_Get_count``

datatype      Informs on the type of the buffer

source        Rank of the *sending* process

tag           A tag for message book-keeping

status        MPI_STATUS_IGNORE for now. See ``MPI_Get_count``.

comm          The communicator!

return        Error tag
----------    -----------------------------------------------------------------------

### Example: Blocking synchronous example

Inside a new section in the test framework:

{% fragment send, cpp/point2point.cc %}

Common bug: Set both sender and receiver to 0. What happens?

### Example: Do you know your C vs C++ strings?

Why the ``+1``?

``` cpp
int const error = MPI_Ssend(
  (void*) peace.c_str(), peace.size() + 1, MPI_CHAR, 1, 42, MPI_COMM_WORLD);
```

. . .

Because C and C++ ``char const*`` strings are null-terminated to indicate the
string is finished, which adds an extra character. However, ``std::string``
abstracts it away. And so its lenght does *not* include the null-termination.

### Example: Causing a dead-lock

Watch out for order of send and receive!

Bad:

``` cpp
if(rank == 0) {
   MPI_Ssend (sendbuf, count, MPI_INT, 1, tag, comm);
   MPI_Recv (recvbuf, count, MPI_INT, 1, tag, comm, &status);
} else {
   MPI_Ssend (sendbuf, count, MPI_INT, 0, tag, comm);
   MPI_Recv (recvbuf, count, MPI_INT, 0, tag, comm, &status);
}
```

Good:

```
if(rank == 0) {
   MPI_Ssend (sendbuf, count, MPI_INT, 1, tag, comm);
   MPI_Recv (recvbuf, count, MPI_INT, 1, tag, comm, &status);
} else {
   MPI_Recv (recvbuf, count, MPI_INT, 0, tag, comm, &status);
   MPI_Ssend (sendbuf, count, MPI_INT, 0, tag, comm);
}
```



### All(most all) point to point

Sending messages:

-----------   -------------  ----------------
name          Blocking       returns before
              (buffer-safe)  message arrival
-----------   -------------  ----------------
MPI_Ssend     yes            no

MPI_Send      yes            maybe

MPI_Isend     no             yes
-----------   -------------  ----------------


Receiving messages:

-----------   ---------
name          blocking
-----------   ---------
MPI_Recv      yes

MPI_Irecv     no
-----------   ---------
