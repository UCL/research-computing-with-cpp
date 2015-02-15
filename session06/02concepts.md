---
title: Aspects of Message 
---

## Concepts

### One to one message passing

Can you think of two behaviors for message passing?

![](session06/figures/mpi.png)

- Process 0 can (i) gives message, (ii) leave, and/or (iii) wait for
  acknowledgements
- Process 1 can (i) receives message
- MPI can (i) receive message, (ii) deliver message, (iii) deliver
  acknowledgments

### Blocking synchronous algorithm

<style>
img {
    position: relative;
    bottom: -20px;
}
</style>
------------------------------   ----------------------------
a. 0, 1, and MPI stand ready:    ![](session06/figures/sync0)
b. message dropped off by 0:     ![](session06/figures/sync1)
c. transit:                      ![](session06/figures/syncT)
d. message received by 1         ![](session06/figures/syncA)
e. receipt received by 0         ![](session06/figures/syncR)
------------------------------   ----------------------------

### Non-blocking synchronous

------------------------------  -----------------------------
a. 0, 1, and MPI stand ready:   ![](session06/figures/sync0)
b. message dropped off by 0:    ![](session06/figures/sync1)
c. transit, 0 leaves            ![](session06/figures/ssyncT)
d. message received by 1        ![](session06/figures/ssyncA)
------------------------------  -----------------------------

### Non-blocking asynchronous

-------------------------------  -----------------------------
a. 0, 1, and MPI stand ready:    ![](session06/figures/sync0) 
b. 0 leaves message in safebox   ![](session06/figures/async1)
c. transit                       ![](session06/figures/asyncT)
d. message received by 1         ![](session06/figures/asyncA)
e. receipt placed in safebox     ![](session06/figures/asyncR)
-------------------------------  -----------------------------

### Collective Communications

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
