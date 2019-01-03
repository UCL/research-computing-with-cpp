---
title: Intermediate CMake
---

## Intermediate CMake

### What's next?

* Most people learn CMake by pasting snippets from around the web
* As project gets larger, its more complex
* Researchers tend to just "stick with what they have."
* i.e. just keep piling more code into the same file.
* Want to show you a reasonable template project.

### Classroom Exercise 4. (or Homework)

* Build https://github.com/MattClarkson/CMakeCatch2.git
* If open-source, use travis and appveyor from day 1.
* We will go through top-level CMakeLists.txt in class.
* See separate ```Code``` and ```Testing``` folders
* Separate ```Lib``` and ```CommandLineApps``` and ```3rdParty```
* You should focus on
    * Write a good library
    * Unit test it
    * Then it can be called from command line, wrapped in Python, used via GUI.
    

### Classroom Exercise 5. (or Homework)

* Try renaming stuff to create a library of your choice.
* Create a github account, github repo, Appveyor and Travis account
* Try to get your code running on 3 platforms
* If you can, consider using this repo for your research
* Discuss
    * Debug / Release builds
    * Static versus Dynamic
    * declspec import/export
    * Issues with running command line app? Windows/Linux/Mac
    

### Looking forward

In the remainder of this course we cover

* Some compiler options
* Using libraries
* Including libraries in CMake 
* Unit testing
* i.e. How to put together a C++ project
* in addition to actual C++ and HPC
