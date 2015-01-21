---
title: Eigen
---

## Using Eigen

### Introduction

* [Eigen][EigenHome] is:
    * C++ template library, 
    * Linear algebra, matrices, vectors, numerical solvers, etc.
    * ![Overview of features](session03/figures/eigenContents)


### Tutorials
    
Obviously, you can read:

* the existing [manual pages][EigenManual] 
* tutorials ([short][EigenShort], [long][EigenLong]).
* the [Quick Reference][EigenRef]


### Getting started

* Header only, just need ```#include```
* Uses CMake, but that's just for 
    * documentation
    * run unit tests
    * do installation.

    
### C++ Principles

(i.e. why introduce Eigen on this course)
 
* Eigen uses
    * Templates
    * Loop unrolling, traits, template meta programming
 
 
### Matrix Class

* This:
{{cppfrag('03','Eigen/HelloMatrix.cc')}}

* Produces:
{{execute('03','Eigen/HelloMatrix')}}


### Matrix Class Declaration

Matrix Class
```
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
class Matrix
  : public PlainObjectBase<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> >
{
```
So, its templates, [review last weeks lecture][Lecture2].


### Matrix Class Construction

But in documentation:

![Matrix construction](session03/figures/eigenMatrixDynamic)

It took a while but I searched and found:

```
src/Core/util/Constants.h:const int Dynamic = -1;
```

and both fixed and dynamic Matrices come from same template class???

How do they do that? 


### DenseStorage.h - 1

In ```src/Core/DenseStorage.h```:

```
template <typename T, int Size, int MatrixOrArrayOptions,
          int Alignment = (MatrixOrArrayOptions&DontAlign) ? 0
                        : (((Size*sizeof(T))%16)==0) ? 16
                        : 0 >
struct plain_array
{
  T array[Size];
```
So, a ```plain_array``` structure containing a stack allocated array.


### DenseStorage.h - 2

In ```src/Core/DenseStorage.h```:
```
// purely fixed-size matrix
template<typename T, int Size, int _Rows, int _Cols, int _Options> class DenseStorage
{
    internal::plain_array<T,Size,_Options> m_data;
```
There is a default template class for DenseStorage, and specialisation for fixed arrays.

    
### DenseStorage.h - 3

In ```src/Core/DenseStorage.h```:
```
// purely dynamic matrix.
template<typename T, int _Options> class DenseStorage<T, Dynamic, Dynamic, Dynamic, _Options>
{
    T *m_data;
    DenseIndex m_rows;
    DenseIndex m_cols;
```
There is a default template class for DenseStorage, and specialisation for Dynamic arrays.


### Eigen Matrix Summary

* Templated type supports dynamic and fixed arrays seamlessly on stack or heap
* typedef's to make life easier: ```Matrix3d``` = 3 by 3 of double
* Uses TMP to generate generate code at compile time
* Benefit from optimisations such as loop unrolling when using fixed size constant arrays


### Eigen Usage


### Further Reading

* [Short Tutorial][EigenShort]
* [Longer Tutorial][EigenLong]

[EigenHome]: http://eigen.tuxfamily.org
[EigenManual]: http://eigen.tuxfamily.org/dox/index.html
[EigenShort]: http://eigen.tuxfamily.org/dox/GettingStarted.html
[EigenLong]: http://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
[EigenRef]: http://eigen.tuxfamily.org/dox/group__QuickRefPage.html
[Lecture2]: http://development.rc.ucl.ac.uk/training/rcwithcpp/session02/