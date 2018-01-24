---
title: Research Software Engineering with C++
layout: default
slidelink: false
---

##Introduction

In this course, we build on your knowledge of C++ to enable you to work on complex numerical codes for research.
Research software needs to handle complex mathematics quickly, so the course focuses on writing software to exploit the
capabilities of modern supercomputers, accelerator chips, and cloud computing facilities. But research software is also
very complicated, so the course also focuses on techniques which will enable you to build software which is easy for colleagues
to understand and adapt.

##Pre-requisites

* You must understand how to use a Unix style terminal, including commands such as ```ls```, ```cd``` and creating and editing files in a text editor of your choice.
* Detailed prior knowledge of C++, including at least creation of classes, understanding abstraction, encapsulation, inheritance and polymorphism.
* You are required to bring your own laptop to the course as the classrooms we are using do not have desktop computers.
* We have provided [setup](98Installation) instructions for installing the software needed for the course on your computer.
* You must understand how your compiler toolchain works.

* Eligibility: This course is for UCL post-graduate students.

Your C++ knowledge should include:

* Compiling a library
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
    <h3>Intro</h3>
    <ul>
     <li>Course structure and admin</li>
     <li>Git quick-start</li>
     <li>CMake quick-start</li>
     <li>Unit testing</li>
    </ul>
   </td>
   
   <td>
    <h3>Distributed Memory Parallelism</h3>
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
    <h3>Libraries</h3>
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
  
   <td>
    <h3>OpenMPI</h3>
    <ul>
     <li>Examples</li>
    </ul>
   </td>
   
  </tr>
  <tr>
  
   <td>
    <h3>C++</h3>
    <ul>
     <li>Language recap</li>
     <li>Essential reading</li>
     <li>Templates</li>
     <li>Smart Pointers</li>
     <li>Exceptions</li>
     <li>Eigen (Linear Algebra)</li>
     <li>Boost</li>
     <li>RAII pattern</li>
     <li>Program to Interfaces</li>
     <li>Dependency Injection</li>
    </ul>
   </td>
    
   <td>
    <h3>OpenMPI</h3>
    <ul>
     <li>TBC</li>
    </ul>
   </td>
    
  </tr>
  <tr>
  
   <td>
    <h3>Shared Memory Parallelism</h3>
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
   
   <td>
    <h3>Accelerators</h3>
    <ul>
     <li>Concepts</li>
     <li>Trivially parallelisable examples<li>
     <li>Map Reduce</li>
    </ul>
   </td>
   
  </tr>
  <tr>
  
   <td>
    <h3>Binding & Tuning</h3>
    <ul>
     <li>Python Binding</li>
     <li>Profiling</li>
     <li>Valgrind</li>
    </ul>
   </td>
   
   <td>
    <h3>Accelerators</h3>
    <ul>
     <li>Thrust</li>
     <li>ArrayFire</li>
     <li>CUDA</li>
    </ul>
   </td>
   
  </tr>
  
 </tbody>
</table>


Versions
--------

You can find the course notes as HTML via the navigation bar to the left.

The [notes](notes.pdf) are also available in  a printable pdf format.
