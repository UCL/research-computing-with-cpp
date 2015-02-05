---
title: Races, Locks and Critical regions
---

## Races, Locks and Critical regions


### Introduction

In the best of worlds our calculations can be done independently. 
However, even in our simplest examples we saw issues.

* `cout` is not thread safe. Garbage mixed output.
* Needs to use `critical` to merge output.
* Real world examples may be more complicated.
* Incorrectly shared variable leads to random and typically wrong results.

## Race condition

When the result of a calculation depends on the timing between threads. 

* Example: threads writing to same variable. 
* Can be hard to detect.
* May only happen in rare cases
* May only happen on specific platforms
* Or depend on system load from other applications

## Barriers and synchronisation

Typically it is necessary to synchronize threads. Make sure that all threads are 
done with a piece of work before moving on to do something else depending on this.
This is done with a barrier.

* Parallel regions such as `omp for` have an implicit barrier at the end. 
    - Threads wait for the last to finish before moving on.
    - May waste significant amount of time.
    - Will return to look at load balancing later.
    - Sometime there is no need to wait. Disable with `nowait`
* Sometimes you need a barrier where there is no implicit barrier.
    - `#pragma omp barrier` inserts a barrier
    - Don't overuse this. Will cause performance drop.

## Protecting code and variables.

* Critical
    - Only one task can execute at a time
    - Protect non thread safe code
* Single 
    - Only one tread executes this code.
    - The first thread that arrives will execute the code
* Master
    - Similar to singe but uses the master thread
* Atomic
    - Protect a variable by changing it in one step.

## Locks

Sometimes the critical regions etc. are not flexible enough to implement your algorithm.
I.e. need to prevent two different pieces of code from running at the same time.

OpenMP locks is a general way to ensure correctness. A thread tries to set the lock. 
If the lock is not held by any other thread it is successful and free to carry on. 
If not it will wait until the lock becomes unset. Important to remember to unset the lock when done.

Can use multiple locks protecting different resources. Can result in deadlocks if two threads needs both locks. 
One thread holds one lock and the other one holds the other. Both are waiting for a lock to be free. 

## Example

Replace the critical region with a lock. 
In this case there is no gain from using a lock.
{{cppfrag('05','locks/simplelock.cc')}} 


## Advanced

OpenMP implements two types of locks. We have only considered simple locks. Consult the OpenMP specks for nested locks.
