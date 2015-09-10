---
title: Tasks
---

## OpenMP Tasks

### Introduction

* Not all problems are easily expressed as for loops.
* The task construction creates a number of tasks
* The tasks are added to a queue
* Threads take a task from the queue

### Example

Calculate Fibonacci numbers by recursion.

* Only as an example:
    - Hard to get any performance improvement. Usually slower that serial code
    - Inefficient algorithm in any case. Why?
    - Consider limiting the number of tasks. Why?
* Use taskwait to ensure results are done before adding their results together

### Code

{% idio cpp/openmptask/fibdemo.cc %}

{% fragment fibfunction %}

### Main function

{% fragment mainfunction %}

{% endidio %}

Note only one thread initially creates tasks. Tasks are still running in parallel.

### Advanced usage

* Task dependency:
    - Depends on child tasks. `#taskwait`
    - Real cases may be more complicated
    - May need to explicitly set dependency
    - `#pragma omp task depends(in/out/inout:variable)`
    - See OpenMP docs for details
* `taskyield` Allows a task to be suspended in favour of a different task:
    - Could be useful together with locks

### Controlling task generation

* `if(expr)` `expr==false` create an undeferred task
    - Suspend the present task and execute the new task immediately on the same tread
* `final(expr)`  `expr==true` This is the final task
    - All child tasks are included in the present task
* `mergeable`
    - Included and undeferred tasks may be merged into the parent task

May useful to avoid creating to many small tasks. I.e. in our Fibonacci example.
