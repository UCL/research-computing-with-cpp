---
title: Concurrency Problems
---

## Issues with concurrency
Running a program with several concurrent threads of execution can encounter several issues that are not present when running a single thread. These problems can include

* Deadlock: the system is locked, not doing any work and not changing state.
* Livelock: the system is apparently doing _something_, changing state, but no work is being done.
* Resource Starvation: one or more of the threads is unable to perform work, being unable to access a necessary resource.
* Race Condition: a thread ensures an action is possible, but this is not longer the case when the action is attempted, due to another thread.

## Dining Philosophers
A classical computer science analogy to illustrate issues with concurrency and resource contention.

### The set up
Five hungry yet contemplative philosophers are seated around a table. In front of each of them is a bowl of noodles. In between each of them is a single chopstick. The philosophers are holding one of their famous _symposia_, a party for eathing and thinking and drinking. Each philosopher will alternate between thinking and eating. To eat, a philosopher needs to have hold of _both_ chopsticks. Crucially, this means that not all of the philosophers can eat at the same time.

A fuller explanation of the Dining Philosphers problem can be found on [Wikipedia](https://en.wikipedia.org/wiki/Dining_philosophers_problem).

## Deadlock
It is easy for the philosophers to get into what is called a deadlocked state. In the analogy, each philosopher is determined to eat, and picks up the chopstick to their right, and is unwilling to put it down until after they have eaten. They are each left holding exactly one chopstick each, and are also left hungry, each unwilling to put down the chopstick the have hold of.

In a concurrent computer system this same deadlock can also occur. If there are _n_ worker threads, each requesting two of _n_ resources, then they can easily end up in a state where each worker has one of the two resources they need, and no work is done. Because of this, the state is referred to as being locked. The computer system is also not changing state, so this is referred to as a deadlock.

### Try and wait
A simple solution to a deadlock is to implement a try-and-wait algorithm.

For the philosophers, this means trying to eat using both chopsticks. If they are, great!, they get to eat. If not, then put down any chopsticks you may have in hand, wait a period of time and repeat.

For the computer system, the procedure is very similar. Try to acquire the necessary resources, use them if available, otherwise free them for use by others and wait for some period of time before trying again.

In both cases, this breaks the time symmetry of the problem. No longer are all the workers or philosphers in the same state for ever more, something is changing, and work gets done.

## Livelock
Or does it? The try-and-wait algorithm might end up with all of the philosophers picking up on chopstick simultaneously, realising the other is not available, and putting it back dowmn again without eating. They all wait for the same length of time, and try again.

This is called a livelock. Just like a deadlock, it is a situation in which no productive work is being done, hence a 'lock'. However, the state of the philosophers (or computer system) is continually changing, so it looks to an observer or program user that something is happening.

