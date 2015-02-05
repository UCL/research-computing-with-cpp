---
title: Races, Locks and Critical regions
---

## Races, Locks and Critical regions


### Introduction

In the best of worlds our calculations can be done independently. 
However, even in our simplest examples we saw issues.

* `std::cout` is not thread safe. Garbage mixed output.
* Needs to use `critical` to merge output.
* Real world examples may be more complicated.
* Incorrectly shared variable leads to random and typically wrong results.

### Race condition

When the result of a calculation depends on the timing between threads. 

* Example: threads writing to same variable
* Can be hard to detect
* May only happen in rare cases
* May only happen on specific platforms
* Or depend on system load from other applications

### Barriers and synchronisation

Typically it is necessary to synchronize threads. Make sure that all threads are 
done with a piece of work before moving on. Barriers synchronizes threads.

* Parallel regions such as `omp for` have an implicit barrier at the end.
    - Threads wait for the last to finish before moving on
    - May waste significant amount of time
    - We will return to look at load balancing later
    - Sometime there is no need to wait. 
    - Disable implicit barrier with `nowait`
* Sometimes you need a barrier where there is no implicit barrier
    - `#pragma omp barrier` inserts a barrier
    - Don't overuse this. Performance drop

### Protecting code and variables.

* `#pragma omp critical`
    - Only one task can execute at a time
    - Protect non thread safe code
* `#pragma omp single`
    - Only one tread executes this block
    - The first thread that arrives will execute the code
* `#pragma omp master`
    - Similar to singe but uses the master thread
* `#pragma omp atomic`
    - Protect a variable by changing it in one step.

### Mutex locks

Sometimes the critical regions are not flexible enough to implement your algorithm.

* Need to prevent two different pieces of code from running at the same time. 
* Need to lock only a fraction of a large array.

### OpenMP locks
 
OpenMP locks is a general way to manage resources in threads. 

* A thread tries to set the lock. 
* If the lock is not held by any other thread it is successful and free to carry on. 
* If not it will wait until the lock becomes unset. 
* Important to remember to unset the lock when done.
* Might otherwise result in a deadlock. Program hangs. 

### Locks

Sometimes it is useful to lock multiple resources with different locks.

* Use multiple locks protecting different resources. 
* Can result in deadlocks if two threads needs both needs the same locks. 
* One thread holds one lock and the other one holds the other. 
* Both are waiting for a lock to be free. 

### Example

Replace the critical region with a lock. 
In this case there is no real gain from using a lock.
{{cppfrag('05','locks/simplelock.cc')}} 


### Notes 

OpenMP implements two types of locks. We have only considered simple locks. 
Consult the [OpenMP specifications][OpenMPSpecs] for nested locks.
