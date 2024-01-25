---
title: MPI Programming
---

# Introductory MPI 

In this section we'll take a bit of a look at what MPI allows us to do, and how this relates to the models we've just discussed. 

We'll explore MPI by building a very simple MPI program, but also highlight some features of MPI that we don't use directly. (We'll explore more MPI in week 10!)

We'll look at just a handful of features:
- How to initialise the MPI environment for a distributed program. 
- How to get the number of processes. 
- How to get the ID ("rank") of current process. 
- How to send a message. 
- How to receive a message. 
- How to end the MPI environment. 

These basic features can handle most of the programming that we want to do in MPI! 

Feel free to follow along with the steps of building this program. Bear in mind that if you do so in inside inside the dev-container you will need to first go into `devcontainer.json` and comment out the line `"remoteUser": "root"`. This is because we shouldn't run MPI from the root user! (Note that some users who are running a recent version of Ubuntu natively on their linux machine may not be able to use the dev-container properly if not as root, in which case you may need to install MPI locally in order to try the example.) Running MPI locally won't allow you to see the performance benefit of the parallelism or catch all the problems that might occur, but you will be able to check that your basic program compiles and works in the way you might expect. 

## Basic Structure of MPI Program

To start with we need a `#include<mpi.h>` statement.

We initialise the MPI environment using `MPI_Init(int *argc, char **argv)` and terminate it with `MPI_Finalize`. 
- `MPI_Init` takes pointers to two variables; `int * argc` is an argument count and `char **v` is an array of argument values. This is similar to the traditional initialisation of `main` with command line arguments. If you don't need to pass anything you can use `NULL` for both of these. 
- `MPI_Finalize` takes no arguments. It doesn't actually kill the processes, but we can't call any MPI functionality after this point e.g. to send messages. This should essentially be the last thing we do in an MPI program. 

```cpp
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    MPI_Finalize();
}
```

Okay, this program is super interesting yet, but we should be able to compile it. 

## Compiling and Running MPI 

We can compile by using the `mpic++` tool instead of `g++`. For example:

```
mpic++ -o simple_mpi mpi_example.cpp
```

To run an MPI executable we use the `mpirun` command with the option `-np` to set the number of processes, e.g.
```
mpirun -np 4 ./simple_mpi
```

Of course our program can't do anything yet so we need to add some stuff to get it to do something. Let's just as a "Hello World" to it. (In general printing from lots of MPI processes is not a great idea but let's put that aside for now.) 

```cpp
#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    cout << "Hello World!" << endl;

    MPI_Finalize();
}

```

```
$ mpic++ -o simple_mpi mpi_example.cpp 
$ mpirun -np 4 ./simple_mpi
Hello World!
Hello World!
Hello World!
Hello World!
```

Okay, so we have four processes inside the MPI region, and each of them greets us. Unfortunately, we don't know yet which process is saying which hello. More importantly, if we can't identify our processes inside our program, we can't give specialised behaviour to processes (e.g. our Parent process model) or send and receive messages! 

## Identifying a Process

We can use the `MPI_Comm_rank` function to get the "rank" of our process. Rank doesn't actually imply a ranking, i.e. it's not hierarchical, it's just an ID for each process. 
- `MPI_Comm_rank(MPI_Comm comm, int *rank)` takes a communicator (`MPI_Comm`) type and a pointer to an int, and returns an int. It uses the pointer argument to store the rank, and the return integer to report any errors. The MPI environment may terminate your program if there is an error before this value gets returned (you can see more about MPI call return values in the [Open MPI documentation](https://www.open-mpi.org/doc/v4.0/man3/MPI_Comm_rank.3.php) in the Error section of any function).
- The communicator is a way of grouping processes which need to communicate with one another together. `MPI_COMM_WORLD` is a pre-defined variable which contains all processes. You can use [MPI_Comm_split](https://www.open-mpi.org/doc/v4.1/man3/MPI_Comm_split.3.php) to create communicators for sub-groups. This can be useful when using MPI's built in methods for broadcasting to all processes within a communicator, as this means you don't have to do all the messages manually and the MPI broadcast strategy will likely be more efficient. 
    - When broadcasting to a group of $N$ other processes it is not efficient to have one process send all $N$ messages sequentially. Instead, you can broadcast using a tree structure, where the origin process sends the message to 2 other processes, and those two can, in parallel, send the message to two further processes each and so on. This allows for overlap in message sending between processes and this kind of forwarding may need to happen anyway depending on the network topology. You can let MPI handle all this for you if you use appropriate communicators!
- A process can be part of multiple communicators (as all processes are part of MPI_COMM_WORLD then if you have any additional communicators then any processes part of those communicators must be contained in at least two) and may have different "rank" (ID) in each communicator. It's important that you properly manage the identification of processes in each communicator if you want things to work! 

So let's utilise our process ID in our little example:

```cpp
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    printf("Hello World! Kind regards, Process %d\n", process_id);

    MPI_Finalize();
}
```
- We have replaced the call to `cout` with `printf`. This is because calls to `printf` are atomic and calls to `cout` are not. So if we use `cout` in principle our messages can get muddled when we use `<<` to stream data to `cout`. 

```
$ mpirun -np 4 ./simple_mpi
Hello World! Kind regards, Process 1
Hello World! Kind regards, Process 2
Hello World! Kind regards, Process 3
Hello World! Kind regards, Process 0
```

Great, we can identify each process! We'll work towards a simple Parent process model for sorting a list, as in the earlier section. As we've noted this isn't a very practical MPI example as it involves moving a lot of data to do very simple work, but it does illustrate how to use the MPI function calls nicely! We can now (arbitrarily) select Process 0 as our Parent process, which will handle splitting up and re-combining the list. 

In order to communicate it to the other processes though, we'll need to know how many processes there are from _inside our program_, since we don't want that kind of thing hard coded. 

## Getting the Total Number of Processes

We can get the total number of processes in a given communicator using `MPI_Comm_size`. This works basically exactly the same as `MPI_Comm_rank`. 

```cpp
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    printf("Hello World! Kind regards, Process %d of %d\n", process_id, num_proc);

    MPI_Finalize();
}
```

```
$ mpirun -np 4 ./simple_mpi
Hello World! Kind regards, Process 1 of 4
Hello World! Kind regards, Process 2 of 4
Hello World! Kind regards, Process 3 of 4
Hello World! Kind regards, Process 0 of 4
```

Now that we can do that we can divide up some work! Let's say we have a list which ends up on Process 0 but not the other processes; in our simple example we'll just generate a random list, but in practice this might be something like file input which need to be localised to single process. We can check how many processes we have an divide our list accordingly. 

```cpp
#include <mpi.h>
#include <random> 

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    if (process_id == 0)
    {
        const int N = 256;
        std::mt19937_64 rng;
        std::uniform_real_distribution<double> dist(0, 1);
        double master_list[N];
        for(int i = 0; i < N; i++)
        {
            master_list[i] = dist(rng);
        }

        int listSize = N / num_proc; // We are using a multiple of four to avoid dealing with remainders!
    }

    printf("Hello World! Kind regards, Process %d of %d\n", process_id, num_proc);

    MPI_Finalize();
}
```

So process 0 now has a random list, and we've calculate the size of sub-list that should go to each process. But how do we send the sub-list on to the right destination?

## Sending Messages with MPI 

We send a message using [MPI_Send](https://www.open-mpi.org/doc/v4.0/man3/MPI_Send.3.php). This one takes a lot of arguments! 
- `int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)`
    - `buf` is a pointer to the data buffer that you want to send. This could be a simple variable or an array. 
    -  `count` is the number of elements in the buffer. 
    - `datatype` is the type of the elements in the buffer. You can find a translation from MPI datatypes to C++ datatypes [here](https://www.mpi-forum.org/docs/mpi-2.1/mpi21-report-bw/node330.htm). 
        - Together `count` and `datatype` tell the environment how much data needs to be sent. Remember that as a pointer `buf` only points to an address in memory; this could be a variable or the start of an array. `buf` on its own has no information about how much data the thing you're pointing to is made up of. 
    - `destination` is the rank of the process you want to send the message to in the given communicator. 
    - `tag` is an integer identifier which can be used to distinguish between types of messages. This is useful for more complex programs where processes might have multiple things they could request from or send to one another. 
    - `comm` is the communicator that you are sending the message within. 

Okay, so let's try sending to our other processes. Our `comm` is just `MPI_COMM_WORLD`, our datatype is `MPI_DOUBLE`, our `count` is `listSize`, our `tag` is arbitrary (let's just say `0`). The only thing to be careful of is `buf` and `destination`: for each destination process we'll need to move `buf`. 

```cpp
#include <mpi.h>
#include <random> 

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    if (process_id == 0)
    {
        const int N = 256;
        std::mt19937_64 rng;
        std::uniform_real_distribution<double> dist(0, 1);
        double master_list[N];
        for(int i = 0; i < N; i++)
        {
            master_list[i] = dist(rng);
        }

        int listSize = N / num_proc; // We are using a multiple of four to avoid dealing with remainders!

        // Send the list data in messages
        for(int i = 1; i < num_proc; i++)
        {
            double * buffer_start = master_list + listSize*i;
            MPI_Send(buffer_start,
                     listSize,
                     MPI_DOUBLE,
                     i,
                     0,
                     MPI_COMM_WORLD);
        }
    }

    printf("Hello World! Kind regards, Process %d of %d\n", process_id, num_proc);

    MPI_Finalize();
}
```

Now if we do this, we won't be able to tell if anything's happened. Crucially, our other processes can't do anything with the information, because they haven't received it yet! Let's find out how to deal with our information on our other processes. 

## Receiving Messages with MPI 

We need to actively receive information using [MPI_Recv](https://www.open-mpi.org/doc/v3.0/man3/MPI_Recv.3.php). We essentially already know how to do this because the signature of this function is almost identical to `MPI_Send`. 
- `int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)`
- `destination` is replaced by `source`. 
- We have one extra parameter called `status`, of type `MPI_Status`. MPI_Status has some information about the source and tag of the message, which can be useful if we use special values of `source` and `tag`. If we don't need the `status` (because we are listening out for a specific source and tag) then we can use the special value `MPI_STATUS_IGNORE`. 
- The values `MPI_ANY_SOURCE` and `MPI_ANY_TAG` allow us to listen for messages with any source and / or tag. Otherwise, this call will only catch a message which comes from the expected destination and with the correct tag. 
- We still need to know the count and the datatype to work out how much data is being copied over. 
- `buf` now points to the data buffer we'll be copying data _into_ rather than from.

Let's receive our lists and print out the first element of the list we receive. To test this, we'll replace our random numbers with a predictable number for now!

```cpp
#include <mpi.h>
#include <random> 

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    if (process_id == 0)
    {
        const int N = 256;
        std::mt19937_64 rng;
        std::uniform_real_distribution<double> dist(0, 1);
        double master_list[N];
        for(int i = 0; i < N; i++)
        {
            master_list[i] = i; //dist(rng);
        }

        int listSize = N / num_proc; // We are using a multiple of four to avoid dealing with remainders!

        // Send the list data in messages
        for(int i = 1; i < num_proc; i++)
        {
            double * buffer_start = master_list + listSize*i;
            MPI_Send(buffer_start,
                     listSize,
                     MPI_DOUBLE,
                     i,
                     0,
                     MPI_COMM_WORLD);
        }
    }
    else
    {
        int listSize = 256/num_proc; // I am cheating here because I don't want to send another message communicating the size in this simple example. 
        double sub_list[listSize]; // Only needs to be big enough to hold our sub list 
        MPI_Recv(sub_list,
                 listSize,
                 MPI_DOUBLE,
                 0,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        printf("Process %d received a list starting with %f\n", process_id, sub_list[0]);
    }

    MPI_Finalize();
}
```

```
mpirun -np 4 ./simple_mpi
Process 1 received a list starting with 64.000000
Process 2 received a list starting with 128.000000
Process 3 received a list starting with 192.000000
```

So we have now managed to successfully communicate our sublists to other processes! Now we just need to sort them and send them back, and the process 0 can merge them! 

```cpp
#include <mpi.h>
#include <random> 
#include <algorithm> 

using namespace std;

// merge two lists which are stored next to one another in a buffer
void merge(double *buffer, int start1, int size1, int size2)
{
    double *working_buffer = new double[size1+size2];
    int count1 = 0; int count2 = 0;
    int start2 = start1 + size1;
    while((count1 < size1) & (count2 < size2))
    {
        double x1 = buffer[start1 + count1];
        double x2 = buffer[start2 + count2];
        if(x1 < x2)
        {
            working_buffer[count1 + count2] = x1;
            count1++;
        }
        else
        {
            working_buffer[count1 + count2] = x2;
            count2++;
        }
    }

    //Fill buffer with whichever values remain
    for(int i = count1; i < size1; i++)
    {
        working_buffer[i + count2] = buffer[start1 + i];
    }
    for(int i = count2; i < size2; i++)
    {
        working_buffer[count1 + i] = buffer[start2 + i];
    }

    for(int i = 0; i < (size1+size2); i++)
    {
        buffer[i] = working_buffer[i];
    }

    delete[] working_buffer;
}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    if (process_id == 0)
    {
        const int N = 256;
        std::mt19937_64 rng;
        std::uniform_real_distribution<double> dist(0, 1);
        double master_list[N];
        for(int i = 0; i < N; i++)
        {
            master_list[i] = dist(rng);
        }

        int listSize = N / num_proc; // We are using a multiple of four to avoid dealing with remainders!

        // Send the list data in messages
        for(int i = 1; i < num_proc; i++)
        {
            double * buffer_start = master_list + listSize*i;
            MPI_Send(buffer_start,
                     listSize,
                     MPI_DOUBLE,
                     i,
                     0,
                     MPI_COMM_WORLD);
        }
        
        std::sort(master_list, master_list + listSize);

        for(int i = 1; i < num_proc; i++)
        {
            double * buffer_start = master_list + listSize*i; // copy received buffers back into master list
            // we just need the sorted sublists, but we don't care which process is sending them so we'll take 
            // them in whichever order they come using MPI_ANY_SOURCE. The loop makes sure that receive enough
            // messages. 
            MPI_Recv(buffer_start,
                     listSize,
                     MPI_DOUBLE,
                     MPI_ANY_SOURCE,
                     1,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        // Merge all the lists
        // Again we'll cheat this loop a bit by assuming that num_proc and N are both powers of two
        // In real code we would have to deal with things like remainders properly but this example is already quite big!
        for(int i = num_proc; i > 1; i /= 2)
        {
            int subListSize = N / i; 
            for(int j = 0; j < i; j+=2)
            {
                merge(master_list, j*subListSize, subListSize, subListSize);
            }
        }

        printf("Sorted List: ");
        for(int i = 0; i < N; i++)
        {
            printf("%f ", master_list[i]);
        }
        printf("\n");
    }
    else
    {
        int listSize = 256/num_proc; // I am cheating here because I don't want to send another message communicating the size in this simple example. 
        double sub_list[listSize]; // Only needs to be big enough to hold our sub list 
        MPI_Recv(sub_list,
                 listSize,
                 MPI_DOUBLE,
                 0,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        printf("Process %d received a list starting with %f\n", process_id, sub_list[0]);
        
        std::sort(sub_list, sub_list+listSize);
        
        MPI_Send(sub_list,
                 listSize, 
                 MPI_DOUBLE,
                 0,
                 1,
                 MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
```

- You don't need to get too caught up in the details of things like the merge function, it's just there so the example works! The important thing to to look at the two different kinds of logic that get executed for Process 0 and for other processes. 
- After performing the sort each of the other processes returns its sorted list in a message, which I've given the tag `1` to differentiate it although this was not necessary. 
- Process 0 will copy over the data from the other processes in whichever order they arrive, since it doesn't matter for the merge process that they the data is copied back into the same place in the buffer, only that each sublist is sorted. This reduces the possibility of idling. 
- After process 0 has all the data, it can proceed to perform all the merges. This loop is simplified to assume that our number of processes and number of elements are powers of two so that everything fits neatly together, but normally this logic would be a bit more complicated. 
- How would you adapt to the case in the notes where processes communicate pairwise and merge the lists before they get to the parent process? 

My output is as follows:

```
$ mpic++ -o simple_mpi mpi_example.cpp 
$ mpirun -np 4 ./simple_mpi
Process 1 received a list starting with 0.086416
Process 2 received a list starting with 0.832426
Process 3 received a list starting with 0.901857
Sorted List: 0.005706 0.022079 0.022079 0.026552 0.029033 0.029033 0.032402 0.032402 0.063318 0.073619 0.079142 0.079142 0.086416 0.086416 0.087516 0.100668 0.101607 0.101607 0.112180 0.112180 0.116003 0.116882 0.118927 0.118927 0.122479 0.130690 0.138265 0.138265 0.138648 0.138648 0.147311 0.147311 0.186586 0.201725 0.210137 0.210990 0.210990 0.222135 0.222135 0.225448 0.225448 0.236615 0.256984 0.272961 0.272961 0.280540 0.281168 0.284114 0.284114 0.285751 0.297123 0.311976 0.323224 0.338689 0.341087 0.343911 0.343911 0.355726 0.362117 0.362117 0.365701 0.366719 0.366719 0.369665 0.402188 0.422952 0.422952 0.434140 0.434140 0.436551 0.436551 0.455645 0.455645 0.458105 0.466998 0.466998 0.487331 0.487331 0.490911 0.490911 0.491490 0.491490 0.506296 0.519371 0.519371 0.528452 0.528452 0.548245 0.548245 0.567172 0.567179 0.586928 0.586928 0.588669 0.588669 0.589800 0.595384 0.604654 0.604654 0.605491 0.605491 0.611837 0.623808 0.623808 0.627886 0.629748 0.644111 0.644111 0.644393 0.654266 0.668870 0.668870 0.672333 0.672333 0.676285 0.687829 0.687829 0.695038 0.695038 0.711230 0.718358 0.719541 0.725961 0.726120 0.735908 0.735908 0.736445 0.736445 0.746571 0.748943 0.749790 0.759026 0.759026 0.791824 0.791824 0.793390 0.795094 0.807170 0.807170 0.811923 0.827346 0.827346 0.828303 0.828303 0.831381 0.832426 0.836365 0.836365 0.843262 0.849987 0.849987 0.852860 0.853215 0.853215 0.859905 0.859905 0.867071 0.867071 0.868726 0.870392 0.870392 0.875281 0.875281 0.887004 0.888796 0.888796 0.894202 0.904333 0.904333 0.909242 0.909242 0.912689 0.912689 0.934162 0.935147 0.935147 0.951291 0.953564 0.953564 0.956864 0.962750 0.969530 0.969530 0.972835 0.980166 0.980166 0.987307 0.987307 0.988584 0.990865 0.005706 0.026552 0.063318 0.073619 0.087516 0.100668 0.116003 0.116882 0.122479 0.130690 0.186586 0.201725 0.210137 0.236615 0.256984 0.280540 0.281168 0.285751 0.297123 0.311976 0.323224 0.338689 0.341087 0.355726 0.365701 0.369665 0.402188 0.458105 0.506296 0.567172 0.567179 0.589800 0.595384 0.611837 0.627886 0.629748 0.644393 0.654266 0.676285 0.711230 0.718358 0.719541 0.725961 0.726120 0.746571 0.748943 0.749790 0.793390 0.795094 0.811923 0.831381 0.832426 0.843262 0.852860 0.868726 0.887004 0.894202 0.934162 0.951291 0.956864 0.962750 0.972835 0.988584 0.990865 0.994564 0.994564 
```

## Summary

Through this example we have seen how to use the six most important methods that virtually all MPI programs must use:

- `MPI_Init`
- `MPI_Finalize`
- `MPI_Comm_rank`
- `MPI_Comm_size`
- `MPI_Send`
- `MPI_Recv`

Using this handful of calls we can create highly complex systems, although there are many other useful calls like [MPI_Bcast](https://www.open-mpi.org/doc/v4.1/man3/MPI_Bcast.3.php) which broadcasts a message to every process in a communicator and therefore simplifies the programming of some models. 

When we have processes performing different jobs we should refactor this behind function calls so as not to have a large, confusing branching `main` where it is difficult to tell what process you are in! 

MPI involves sending buffers of contiguous memory as messages, and we have used traditional C arrays to align with this interpretation. But we can send C++ datastructures if we need to. We can send an `std::vector v` by using the pointer to the first element `&v[0]` as the buffer pointer. Be wary of sending vectors of objects though; sending generally store these as vectors of pointers and those pointers will not be valid in the memory space of another process! Likewise any objects that you send which contain pointers will not be valid any more either. In general, try to keep you messages short, and composed of a single, simple data type like `char`, `int`, or `double`. 