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

* Detailed prior knowledge of C++
* You are required to bring your own laptop to the course as the classrooms we are using do not have desktop computers.
* We have provided [setup](98Installation) instructions for installing the software needed for the course on
your computer.
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

Members of doctoral training schools, or Masters courses who offer this module as part of their programme should register through their course organisers. Other UCL graduate students or post-doctoral staff can register at [UCL Market Place](http://onlinestore.ucl.ac.uk/) (search for MPHYG002).

Further information on the [UCL EPSRC Centre for Doctoral Training in Medical Imaging website](http://medicalimaging-cdt.ucl.ac.uk/programmes).

This course may not be audited.

##Synopsis

<table>
 <tbody>
  <tr>
   <td>

<h3>Intro</h3><ul>

    <li>Intro, course admin</li>
    <li>Git quick-start</li>
    <li>CMake quick-start</li>
    <li>C++ unit testing framework</li>
    <li>Quick C++ reminder</li>

   </ul></td>
  <td>

<h3>Shared Memory Parallelism</h3><ul>

    <li>OpenMP</li>
    <li>Parallel sections, reduction</li>
    <li>Safety, locks and races</li>
    <li>Task-oriented parallelism</li>
    <li>Load balancing</li>
    <li>OpenMP alternatives</li>

   </ul></td>
  </tr>
  <tr>
   <td>

<h3>Better C++</h3><ul>

   <li>Quick language recap</li>
   <li>Templates</li>
   <li>Error handling</li>
   <li>Construction</li>

   </ul></td>
   <td>

<h3>Distributed Memory Parallelism</h3><ul>

   <li>Concepts</li>
   <li>Point to point communication</li>
   <li>Collective communication</li>
   <li>Groups and communicators</li>
   <li>Advanced communications concepts</li>
   

   </ul></td>
  </tr>
  <tr>
   <td>

<h3>More C++</h3><ul>

  <li>Smart Pointers</li>
  <li>Design Patterns</li>
  <li>C++11/C++14 features</li>
  <li>static asserts</li>

   </ul></td>
   <td>

<h3>MPI Design Example</h3><ul>

   <li>SmoothLife: An example parallel computing problem</li>
   <li>Domain Decomposition</li>
   <li>Local Communication with Sendrecv</li>
   <li>Scripting Job Submission</li>
   <li>Simplifying communication with derived datatypes</li>
   <li>Overlapping Computation and Communication</li>

   </ul></td>
  </tr>
  <tr>
   <td>

<h3>Data Structures - STL</h3><ul>

   <li>TBC</li>

   </ul></td>
   <td>

<h3>Accelerators</h3><ul>

   <li>Introduction to Accelerators</li>
   <li>Using a GPU as an Accelerator</li>
   <li>Using GPU-accelerated libraries</li>
   <li>Using CUDA with Thrust</li>

   </ul></td>
  </tr>

  <tr>
   <td>

<h3>Libraries</h3><ul>

    <li>Using libraries</li>
    <li>Linear algebra (Eigen)</li>
    <li>Boost</li>
    <li>Memory, processors, cores and caches</li>
    <li>Amdahl's law</li>
    <li>How parallel computers work</li>
    <li>High performance programming</li>
    <li>Schedulers and job submission</li>

   </ul></td>
   <td>

<h3>Cloud computing and big data</h3><ul>

   <li>'Big data'</li>
   <li>Working in the cloud</li>
   <li>Virtualisation</li>
   <li>Distributed computing</li>
   <li>Hadoop and MapReduce</li>

   </ul></td>
  </tr>

 </tbody>
</table>


Versions
--------

You can find the course notes as HTML via the navigation bar to the left.

The [notes](notes.pdf) are also available in  a printable pdf format.
