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

Members of doctoral training schools who offer this module as part of their programme should register through their course organisers. Other graduate students who wish to register should  send a 2-page CV to Rebecca Holmes ([rebecca.holmes@ucl.ac.uk](mailto:rebecca.holmes@ucl.ac.uk)), Centre for Doctoral Training in Medical Imaging Administrator.

Further information on the [UCL EPSRC Centre for Doctoral Training in Medical Imaging website](http://www.ucl.ac.uk/imaging-cdt/ProgrammeStructure/accordian/MPHYG001).

##Synopsis

<table>
 <tbody>
  <tr>
   <td>

<h3>Effective C++</h3><ul>

<li>C++ concepts recap</li>
<li>Developer tips</li>
<li>Building with CMake</li>
<li>Unit Testing C++</li>

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

<h3>Templates</h3><ul>

   <li>Introduction to templates</li>
   <li>Function templates</li>
   <li>Class templates</li>
   <li>Template metaprogramming</li>

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

<h3>libraries</h3><ul>

   <li>Working with Libraries</li>
   <li>EIGEN and Linear Algebra libraries</li>
   <li>Boost</li>
   <li>ITK</li>

   </ul></td>
   <td>

<h3>Input and Output for MPI Programs</h3><ul>

   <li>Write in C++, visualise dynamically</li>
   <li>Problems with text output</li>
   <li>Basic binary output</li>
   <li>Endianness and portability</li>
   <li>Problems with multiple output files</li>
   <li>Writing from rank zero</li>
   <li>Problems with serial IO</li>
   <li>MPI-IO</li>

   </ul></td>
  </tr>
  <tr>
   <td>

<h3>High Performance computing</h3><ul>

   <li>The story of HPC</li>
   <li>Parallel computing concepts</li>
   <li>How to use a supercomputer</li>


   </ul></td>
   <td>

<h3>Accelerators</h3><ul>

   <li>Introduction to Accelerators</li>
   <li>Using a GPU as an Accelerator</li>
   <li>Using GPU-accelerated libraries</li>
   <li>Using CUDA with Thrust</li>
   <li>OpenACC</li>
   <li>Writing your own GPU code using CUDA-C</li>

   </ul></td>
  </tr>

  <tr>
   <td>

<h3>Shared memory parallelism</h3><ul>

      <li>OpenMP</li>
      <li>Parallel sections, reduction</li>
      <li>Safety, locks and races</li>
      <li>Task-oriented parallelism</li>
      <li>Load balancing</li>
      <li>OpenMP alternatives</li>

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
