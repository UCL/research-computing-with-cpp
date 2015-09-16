---
title: Research Software Engineering with C++
layout: default
---

##Introduction

In this course, we build on your knowledge of C++ to enable you to work on complex numerical codes for research.
Research software needs to handle complex mathematics quickly, so the course focuses on writing software to exploit the
capabilities of modern supercomputers, accelerator chips, and cloud computing facilities. But research software is also
very complicated, so the course also focuses on techniques which will enable you to build software which is easy for colleagues
to understand and adapt.

##Pre-requisites

* Detailed prior knowledge of C++
* You are required to bring your own laptop to the course as the classrooms we are using do not have desktop computers.
* We have provided [setup](99installation) instructions for installing the software needed for the course on
your computer.
* Eligibility: This course is for UCL post-graduate students.

Your C++ knowledge should include:

* Compiling a library, testing debugging
* Arrays
* Structures, dynamically allocated arrays
* Classes
* Operator overloading
* Inheritance and polymorphism

This could be obtained via a variety of C++ courses in college, such as
[MPHYGB24](https://moodle.ucl.ac.uk/course/view.php?id=5395) <!-- or the RITS introductory C++ course -->
or through online resources such as [UCL Lynda](https://www.ucl.ac.uk/lynda),
 e.g. http://www.lynda.com/C-tutorials/C-Essential-Training/182674-2.html

##Registration

Members of doctoral training schools who offer this module as part of their programme should register through their course organisers. Other graduate students who wish to register should  send a 2-page CV to Rebecca Holmes ([rebecca.holmes@ucl.ac.uk](mailto:rebecca.holmes@ucl.ac.uk)), Centre for Doctoral Training in Medical Imaging Administrator.

Further information on the [UCL EPSRC Centre for Doctoral Training in Medical Imaging website](http://www.ucl.ac.uk/imaging-cdt/ProgrammeStructure/accordian/MPHYG001).

##Synopsis

<table>
 <tbody>
  <tr>
   <td>

### Effective C++

* C++ concepts recap
* Developer tips
* Building with CMake
* Unit Testing C++

   </td>
   <td>

### Distributed Memory Parallelism

   * Concepts
   * Point to point communication
   * Collective communication
   * Groups and communicators
   * Advanced communications concepts

   </td>
  </tr>
  <tr>
   <td>

### Templates

   * Introduction to templates
   * Function templates
   * Class templates
   * Template metaprogramming

   </td>
   <td>

### MPI Design Example

   * SmoothLife: An example parallel computing problem
   * Domain Decomposition
   * Local Communication with Sendrecv
   * Scripting Job Submission
   * Simplifying communication with derived datatypes
   * Overlapping Computation and Communication

   </td>
  </tr>
  <tr>
   <td>

### libraries

   * Working with Libraries
   * EIGEN and Linear Algebra libraries
   * Boost
   * ITK

   </td>
   <td>

### Input and Output for MPI Programs

   * Write in C++, visualise dynamically
   * Problems with text output
   * Basic binary output
   * Endianness and portability
   * Problems with multiple output files
   * Writing from rank zero
   * Problems with serial IO
   * MPI-IO

   </td>
  </tr>
  <tr>
   <td>

### High Performance computing

   * The story of HPC
   * Parallel computing concepts
   * How to use a supercomputer


   </td>
   <td>

### Accelerators

   * Introduction to Accelerators
   * Using a GPU as an Accelerator
   * Using GPU-accelerated libraries
   * Using CUDA with Thrust
   * OpenACC
   * Writing your own GPU code using CUDA-C

   </td>
  </tr>

  <tr>
   <td>

### Shared memory parallelism

      * OpenMP
      * Parallel sections, reduction
      * Safety, locks and races
      * Task-oriented parallelism
      * Load balancing
      * OpenMP alternatives

   </td>
   <td>

### Cloud computing and big data

   - 'Big data'
   - Working in the cloud
   - Virtualisation
   - Distributed computing
   - Hadoop and MapReduce

   </td>
  </tr>

 </tbody>
</table>


Versions
--------

You can find the course notes as HTML via the navigation bar to the left.

The [notes](notes.pdf) are also available in  a printable pdf format.
