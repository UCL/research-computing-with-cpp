---
title: Asynchronous MPI Programs
---

# Asynchronous Strategies

Last week we focussed our attention on distributed programs in which each process completes an equal amount of work in a roughly equal period of time, and then shares its results either with all the other process or with a master process. This kind of parallelism in which each process proceeds in lock-step with one another is called _synchronous_; by contrast an _asynchronous_ program allows the processes to proceed more independently without barriers which force all of the processes to be kept closely in sync. 

Asynchronous strategies are particularly useful when the amount of time that a process will take to produce a result that needs to be shared is unpredictable. Consider the following simple problem of cracking a password by brute force:

- For an $n$-bit key there are $2^n$ possible keys
- Each possible key is tried until a key is found which works to decrypt the message / access the resource etc.

This brute force approach is easily parallelised by assigning each process a subset of the possible keys to check (so that no process repeats the work of another process). Here are two possible synchronous approaches to the problem:

1. Frequent Updates:
    - Each process tries one key and determines success or failure.
    - Each process shares its result with the others and if any process is successful then the key is sent to the master process and all other processes stop.
2. Update once at the end:
    - Each process tries each key in its subset and determines success or failure each time. 
    - A process finishes its work when it finds the correct key or exhausts its subset of keys. 
    - Each process sends a message to the master process when it finishes its work. 
    - In general the master process has to wait for all processes to finish before it can receive messages since it doesn't know which process will finish first, and therefore doesn't know which message to check first.

We can see that both of these approaches are sub-optimal in different ways. The first allows for early termination once the key has been found, but wastes enormous amounts of time sending messages with no useful information since every process must report its failures, and this reporting blocks each of the processes from continuing their work until it has been determined whether the key has been found. The second approach avoids all this messsage sending wasting time on each process, but it is disastrous because most processes are doomed to fail (the key will only be found in one subset i.e. one process) and waiting for these to finish means waiting for an _exponential time_ algorithm to complete. 

An _asynchronous_ approach allows us to take advantage of the early termination of the first strategy, and the minimal messaging of the second strategy. 

The asynchronous approach is straight-forward to conceptualise in human terms:

- Each process needs to check through its subsets of keys one at a time.
- If a process finds the key, then it gives the keys to the master process and all other processes are told to stop.

The problem here is that processes need a way of knowing that another process has sent it a message. Using `MPI_Recv` means that our process sits around _waiting for a matching message_, which we don't want to do since we don't know _which_ process will send us a message or when, so we are wasting valuable time and might be waiting for a message that never comes. 

If we know that a process _might_ receive a message, then we need to regularly _check_ whether a message has arrived or not. In a message has arrived, we can read it and act upon it, and if not we can continue to do something else while we wait. 

We can use this idea to make our asynchronous algorithm more explicit

- Each process loops over the keys in its subset 
- The master process loop: 
    - the master process checks its current key
    - if the master process finds the key then it sends a message to the worker processes to stop.
    - otherwise it checks to see if any of the other processes have sent it a message
    - if they have then it stores the received key and sends a message to the other worker process to stop
    - if no processes have sent it a message it generates the next key and returns to the start of the loop
- The worker process loop:
    - the worker process checks its current key
    - if the worker process finds the key then it sends a message to the master process with the key in it and can then stop.
    - otherwise it checks to see if the master has sent it a message to stop
    - if it is not told to stop then it generates the next key and returns to the beginning of the loop

Note that _checking_ if a message has arrived does not require it to wait for a message: if nothing has arrived it moves right along. Also note that even though each process is going through a similar loop, **each process' loop is independent of all the others**. Processes can move at very different speeds and they are not required to synchronise at each iteration.

Checking for messages like this can feel a bit clunky, but it is necessary for any process to be aware of what is being sent to it. If you are working on an asynchronous design it's important to check regularly for updates to avoid processes wasting work on outdated information or a task which has already been completed elsewhere.

## MPI Syntax For Asynchronous Methods

In order to implement this kind of asynchronous strategy we need to introduce some new MPI functions. In order to check for a message we use the `MPI_Probe` function. The arguments are very similar to `MPI_Recv`, although they are in a different order and you don't need to know size and type of the data being sent/

- `int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag MPI_Status *status)`
    - `source`, `tag`, `comm`, and `MPI_Status` function that same as `MPI_Recv`.
    - `flag` is the really crucial part of this function. This is a **pointer** to an `int` value which will be modified by the function. If a message with a matching source and tag is found then `*flag` will evaluate to `1`, and otherwise `*flag` will evaluate to `0`. 

We might implement this inside a loop as follows:

```
while(!complete)
{
    // do work for this iteration
    ...

    // check for messages from other ranks
    for(int &rank : process_ranks)
    {
        int message_found = 0;
        MPI_Iprobe(rank, 0, MPI_COMM_WORLD, &message_found, MPI_STATUS_IGNORE);
        if(message_found)
        {
            complete = true;
            break;
        }
    }
}

// terminate processes and clean up
...
```

# Optional: Asynchronous Time Dependent Simulations

Oftentimes in the sciences we are working on _simulations_, typically updating a time-dependent system by some time-step in a loop. We can parallelise simulations by dividing the domain of the simulation between processes, as in the example of diving a grid from last week. In simulations however, information often needs to be communicated across the boundaries of the sub-domains: think, for example, of a moving particle crossing from one quadrant of space to another. In order for our simulation to be consistent we need to make sure that the information which is communicated from one sub-domain to the other happens consistently *in time*: if the particle leaves one quadrant at time $t$ then it also needs to enter the other quadrant at time $t$. This is automatically the case in a synchronous simulation with a barrier at the end of each iteration to exchange information and update the global clock. But what about an _asynchonrous_ simulation? In this case different processes -- different sub-domains of the simulation -- could have reached different simulation times $t_p$ for each process $p$. How can we communicate information across a boundary like this?

Let's assume that information must pass from process $p$ to $q$, i.e. $p$ is sending some information which will affect subsequent simulation on $q$. There are three cases to consider:
1. $t_p = t_q$: the information can be updated just as in the synchronous simulation.
2. $t_p > t_q$: process $p$ is ahead of process $q$, so $q$ is not yet ready to receive it. In this case the information can be stored until $q$ catches up and then the update can take place. Process $p$ can continue simulating as normal while this is happening so the simulation is not blocked.
3. $t_p < t_q$: process $q$ is ahead of process $p$. This is the most challenging case because all timesteps that $q$ has calculated which are $> t_p$ are now invalidated. In this case we have wasted some work, and we must also be able to **back track** process $q$ back to time $t_p$, then perform the update using the information from process $p$ and restart the simulation on $q$. This approach requires the simulation to be reversible in some way. 
    - Simluations typically can't be reversed analytically in this way and so this may require storing some kind of state history, which can use significant memory. 
    - Depending on your balance of resources, you may only been able to store occasional snapshots of the state and therefore have to backtrack _further_ than $t_p$ and evolve forward to $t_p$ again before being able to perform the update. 
    - You can also store the _changes_ to the state rather than the state itself for each time step (like a git commit!), and roll back these changes in order to reach a previous state.
    - The best strategy will be strongly dependent on your specific simulation problem and the computing resources available to you.
