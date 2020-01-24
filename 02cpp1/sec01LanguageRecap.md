---
title: C++ language recap 
---

## Recap of C++ features

### C++ is evolving 

* C++98
   * Introduced templates
   * STL containers and algorithms
   * Strings and IO/streams  
* C++11
   * Many new features introduced, feels like a different programming language
   * Move semantics
   * `auto`
   * Lambda functions
   * `constexpr`
   * Smart pointers 
   * `std::array`
   * Support for multithreading
   * Regular expressions
* C++14
   * `auto` works in more places
   * Generalised lambda functions
   * `std::make_unique`
   * Reader/writer locks 
* C++17
   * fold expressions
   * `std::any` and `std::variant`
   * The Filesystem library
   * and more...   

C++ constantly evolving, you don't just learn once and then stop you need keep up with language developments. For this course we will be using up until C++14.  

### Some excercises using   

* Now some excercises manipulating `vectors`, using range based for loops and some some algorithms from `<algorithm>`.  

### Homework - 9

* Use [https://github.com/MattClarkson/CMakeLibraryAndApp.git](https://github.com/MattClarkson/CMakeLibraryAndApp.git) and create a new library to perform various operations on a std::vector<int>
   * Using a range base for loop Write a function that prints all the elements on the vector to screen and call this from `myApp`
   * Write a function using a range based for loop that counts the number of elements equal to a value 5
   * Now repeat but instead use the `std::count` algorithm from the `<algorithm>` library 

### Homework - 10

* Again using [https://github.com/MattClarkson/CMakeLibraryAndApp.git](https://github.com/MattClarkson/CMakeLibraryAndApp.git)
   * Write a function `add_elements(vector<int> v, int val, int ntimes)` that takes a `vector<int>` and appends `ntimes` new elements with value `val`  
   * Print contents of the input vector to screen before and after calling the function as well from within the `add_elements` function, what is the problem?
   * Try passing by reference `add_elements(vector<int> &v, ...` instead, does it work now? 
   * What are the advantages/disadvantages to passing by reference? 
   * Try passing as a `const` reference `add_elements(const vector<int> &v, ...`, what happens now and why would you use this?     
