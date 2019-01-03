---
title: Research Software Engineering with C++
layout: default
slidelink: false
---

##Introduction

In this course, we build on your knowledge of C++ to enable you to work on complex numerical codes for research.
Research software needs to handle complex mathematics quickly, so the course focuses on writing software to perform multi-threaded computation. But research software is also
very complicated, so the course also focuses on techniques which will enable you to build software which is easy for colleagues
to understand and adapt.

##Pre-requisites

* For 2019, we will start the course in a UCL Cluster (Computer) Room. However, you
  should bring your own laptop to the course and try to get all the C++ working on your own environment.
* Prior knowledge of C++, see below.
* You should understand how your compiler toolchain works.
* You must also understand how to use a Unix style terminal, including commands such as ```ls```, ```cd``` and creating and editing files in a text editor of your choice.
* We have provided [setup](98Installation) instructions for installing the software needed for the course on your computer.

* Eligibility: This course is for UCL post-graduate students.

Your C++ knowledge should include:

* Compiling a library, using CMake.
* Testing and debugging
* Arrays and structures
* Containers
* Classes
* Operator overloading
* Inheritance and polymorphism

This could be obtained via a variety of C++ courses in college, such as
[MPHYGB24](https://moodle.ucl.ac.uk/course/view.php?id=5395)
or through online resources such as [UCL Lynda](https://www.ucl.ac.uk/lynda),
 e.g. [http://www.lynda.com/C-tutorials/C-Essential-Training/182674-2.html](http://www.lynda.com/C-tutorials/C-Essential-Training/182674-2.html)

##Registration

Members of doctoral training schools, or Masters courses who offer this module as part of their programme should register through their course organisers.

Further information on the [UCL EPSRC Centre for Doctoral Training in Medical Imaging website](http://medicalimaging-cdt.ucl.ac.uk/programmes).

This course may not be audited without the prior permission of the course organiser Dr. Matt Clarkson.

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
     <li>Building C++</li>        
    </ul>
   </td>

  </tr>
  <tr>
  
   <td>
    <h3>2. C++ Libraries</h3>
    <ul>
     <li>Why use libraries?</li>
     <li>Choosing libraries</li>
     <li>How to include libraries</li>
     <li>Finding libraries with CMake</li>
     <li>Module mode</li>
     <li>Config mode</li>
     <li>Examples</li>
     <li>Using non-CMake'd libraries</li>     
    </ul>
   </td>
  
  </tr>
  <tr>
  
   <td>
    <h3>3. C++ and TDD</h3>
    <ul>
     <li>Passing data to functions</li>
     <li>Functions Vs Objects</li>
     <li>Object Oriented Design</li>
     <li>Unit testing</li> 
     <li>Test Driven Development</li>
    </ul>
   </td>

  </tr>
  <tr>
     
   <td>
    <h3>4. More C++</h3>
    <ul>
     <li>Essential reading</li>
     <li>Templates</li>
     <li>Smart Pointers</li>
     <li>Exceptions</li>
     <li>RAII pattern</li>
     <li>Program to Interfaces</li>
     <li>Dependency Injection</li>
    </ul>
   </td>

  </tr>
  <tr>

   <td>
    <h3>5. Debugging, Optimisating for C++</h3>
    <ul>
     <li>Debugging</li>
     <li>Valgrind</li>
     <li>Profiling</li>
    </ul>
   </td>
  
  </tr>
  <tr>
  
   <td>
    <h3>6. Shared Memory Parallelism</h3>
    <ul>
     <li>OpenMP</li>
     <li>Memory, processors, cores and caches</li>
     <li>Amdahl's law</li>
     <li>How parallel computers work</li>
     <li>High performance programming</li>
     <li>Schedulers and job submission</li>     
     <li>Parallel sections, reduction</li>
     <li>Safety, locks and races</li>
     <li>Task-oriented parallelism</li>
     <li>Load balancing</li>
     <li>OpenMP alternatives</li>
    </ul>
   </td>
     
  </tr>
  <tr>
         
   <td>
    <h3>7. Distributed Memory Parallelism</h3>
    <ul>
     <li>Concepts</li>
     <li>Point to point communication</li>
     <li>Collective communication</li>
     <li>Groups and communicators</li>
     <li>Advanced communications concepts</li>
    </ul>
   </td>

  </tr>
  <tr>


   <td>
    <h3>8. Distributed Memory Parallelism Continued</h3>
    <ul>
     <li>Running multiple nodes</li>
     <li>Unit testing</li>
     <li>Examples</li>
    </ul>
   </td>

  </tr>
  <tr>


   <td>
    <h3>9. Python Binding</h3>
    <ul>
     <li>Boost Python</li>
     <li>PyBind11</li>
    </ul>
   </td>

  </tr>
  <tr>


   <td>
    <h3>10. Contingency</h3>
    <ul>
     <li>Depends on how fast the previous lectures go.</li>
    </ul>
   </td>

  </tr>

 </tbody>
</table>


Versions
--------

You can find the course notes as HTML via the navigation bar to the left.

The [notes](notes.pdf) are also available in  a printable pdf format.
