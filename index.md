---
title: Research Software Engineering with C++
layout: default
slidelink: false
---

## Introduction

In this course, we build on your knowledge of C++ to enable you to work on complex numerical codes for research.
Research software needs to handle complex mathematics quickly, so the course focuses on writing software to perform multi-threaded computation. But research software is also
very complicated, so the course also focuses on techniques which will enable you to build software which is easy for colleagues
to understand and adapt.

## Pre-requisites

* Prior knowledge of C++, see below.
* You should understand how your compiler toolchain works.
* You must also understand how to use a Unix style terminal, including commands such as ```ls```, ```cd``` and creating and editing files in a text editor of your choice.
* We have provided [setup](98Installation) instructions for installing the software needed for the course on your computer.

* Eligibility: This course designed for UCL post-graduate students but with agreement of their course tutor a limited number of undegraduate students can also take it.

Your C++ knowledge should include:

* Compiling a library, using CMake.
* Testing and debugging
* Arrays and structures
* Containers
* Classes
* Operator overloading
* Inheritance and polymorphism

This could be obtained through online resources such as the the C++ Essential Training course by Bill Weinman on [LinkedIn Learning](https://www.ucl.ac.uk/isd/linkedin-learning) (accessable using your UCL single sign-on) or via a variety of C++ courses in college, such as [MPHYGB24](https://moodle.ucl.ac.uk).

## Registration

Members of doctoral training schools, or Masters courses who offer this module as part of their programme should register through their course organisers.

This course may not be audited without the prior permission of the course organiser Dr. Jim Dobson as due to the practical nature of the lectures there is a cap on the total number of students who can enroll. 

##Synopsis

This year we propose to cover the following topics. These notes will be updated as we go through the course.

<table>
 <tbody>

  <tr>

   <td>
    <h3>1. Intro</h3>
    <ul>
     <li>Course structure and admin</li>
     <li>CMake</li>
     <li>Unit tests</li>        
     <li>Getting setup with the development environment</li>        
    </ul>
   </td>

  </tr>
  <tr>
  
   <td>
    <h3>2. Modern C++ (1)</h3>
    <ul>
     <li>Recap of features and concepts</li>
     <li>OO concepts: encapsulation and data abstraction, inheritance, polymorphism</li>
    </ul>
   </td>
  
  </tr>
  <tr>
  
   <td>
    <h3>3. Modern C++ (2)</h3>
    <ul>
     <li>C++ standard library</li>
     <li>Smart pointers</li>
     <li>Lambda constructs</li>
    </ul>
   </td>

  </tr>
  <tr>
     
   <td>
    <h3>4. Modern C++ (3)</h3>
    <ul>
     <li>Extensible code design</li>
     <li>Patterns</li>
     <li>Templates</li>
    </ul>
   </td>

  </tr>
  <tr>

   <td>
    <h3>5. Libraries for research computing</h3>
    <ul>
     <li>Including libraries</li>
     <li>Boost</li>
     <li>Linear algebra packages</li>
    </ul>
   </td>
  
  </tr>
  <tr>
  
   <td>
    <h3>6. Code Quality</h3>
    <ul>
     <li>Tools for pretty code</li>
     <li>Memory leaks</li>
     <li>Profiling</li>
     <li>Benchmarking</li>
    </ul>
   </td>
     
  </tr>
  <tr>
         
   <td>
    <h3>7. Performance programming</h3>
    <ul>
     <li>What a computer is and how one works</li>
     <li>Memory, processors, and cores</li>
     <li>Caches</li>
     <li>How parallel computers work</li>
     <li>Shared and distributed memory</li>
     <li>Schedulers and job submission</li>
    </ul>
   </td>

  </tr>
  <tr>


   <td>
    <h3>8. Shared memory parallelism (1)</h3>
    <ul>
     <li>OpenMP</li>
     <li>Parallel sections, Reduction</li>
     <li>Safety, Locks and races</li>
    </ul>
   </td>

  </tr>
  <tr>


   <td>
    <h3>9. Shared memory parallelism (2)</h3>
    <ul>
     <li>Task-oriented parallelism</li>
     <li>Load balancing</li>
    </ul>
   </td>

  </tr>
  <tr>


   <td>
    <h3>10. Distributed memory parallelism</h3>
    <ul>
     <li>Message passing, MPI</li>
     <li>Parallel storage</li>
     <li>Parallel algorithms and design</li>
    </ul>
   </td>

  </tr>

 </tbody>
</table>


Versions
--------

You can find the course notes as HTML via the navigation bar to the left.

The [notes](notes.pdf) are also available in  a printable pdf format.
