---
title: Using Compiler Directives
---

## Using Compiler Directives

### OpenACC

* OpenACC is a set of compiler directives that execute blocks of code on an accelerator
    - not specific to GPUs
    - also works with APU (AMD), Xeon Phi (Intel), etc.
* Set of compiler ```#pragma```s to specify:
    - blocks of code to be run on an accelerator
    - data movement between host and accelerator
* Similar to OpenMP
    - doesn't modify existing code
    - is ignored by compilers that don't support it
* Requires compiler support
    - pgic (non-free, activate with ```-acc``` flag)
    - gcc support coming

### OpenACC Pragmas

* ```#pragma acc kernels``` specifies a block of code to be run in parallel on an accelerator
    - uses a lot of autodetection
* ```#pragma acc data``` specifies data movement between host and accelerator
    - can be used to control data movement between kernel calls
* Each pragma can be customised with clauses that appear at the end of the line
    - ```#pragma acc <pragma> [clause...]```

### OpenACC Pragma Clauses

* ```if(condition)```
    - executes only if the condition evaluates to true
    - falls back to CPU code otherwise
    - can be applied to both kernels and data pragmas among others

### Clauses specific to pragma acc kernel

* ```num_gangs(n)```
    - specifies the number of thread blocks to use
* ```num_workers(n)```
    - specifies the number of threads to use in each block
* ```reduction(op, val)```
    - performs a reduction using the specified operator and initial value
    - similar to OpenMP's ```reduce``` pragma

### General clauses for data movement

* ```copy(var1,var2,...)```
    - allocates and copies variables from the host to the accelerator before a block
    - copies variables back to the host and deallocates after a block
* ```copyin(var1,var2,...)``` and ```copyout(var1,var2,...)```
    - copies data onto the accelerator at the start of a block or off the accelerator at the end of a block
* ```create(var1,var2,...)```, ```delete(var1,var2,...)```
    - allocates variables on the accelerator at the start of a block and deallocates them at the end
    - useful for temporary arrays
* ```present(var1,var2,...)```
    - variable is already present on device so don't allocate or copy
    - also available as ```present_or_copy```, ```present_or_create```, ```present_or_copyin```, ```present_or_copyout```
* ```private(var1,var2,...)```
    - variable is copied to each thread (similarly to OpenMP's ```private```)

### OpenACC SAXPY

* The following code snippet implements a SAXPY kernel using OpenACC:
{{cppfrag('09','saxpy/openacc.c','saxpy')}}
* The only pragma required is the ```kernels``` pragma which turns the ```for``` loop into a GPU kernel

### OpenACC SGEMM

* OpenACC can also be used to implement an SGEMM kernel:
{{cppfrag('09','sgemm/openacc.c','sgemm')}}

### Further Information

* Further information (documentation/tutorials) is available on the (OpenACC website)[http://www.openacc-standard.org/]
