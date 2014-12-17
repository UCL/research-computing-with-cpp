---
title: Function Templates
---

## Function Templates

### Example

Credit to [this](http://www.cplusplus.com/doc/tutorial/functions2/).

{{cppfrag('02','sumFunctionExample/sumFunctionExample.cc')}}

* Produces this output when run:

{{execute('02','sumFunctionExample/sumFunctionExample')}}

### Why

See also [this](http://www.codeproject.com/Articles/257589/An-Idiots-Guide-to-Cplusplus-Templates-Part).

* Reduce your code duplication

```c++
int Add(int a, int b);
double Add(double a, double b);
```

* Reduce your maintenance
* Reduce your effort

### Language Definition

See also [this](http://en.cppreference.com/w/cpp/language/function_template).

Given
```
template < parameter-list > function-declaration
```
so
```
template < class T >  // note 'class'
void MyFunction(T a, T b) 
{
  // do something
}
```
or
```
template < typename T1, typename T2 >  // note 'typename'
T1 MyFunctionTwoArgs(T1 a, T2 b) 
{
  // do something
}
```

Also

* Can use ```class``` or ```typename```.
* I prefer ```typename```.
* Template parameter can apply to references, pointers, return types, arrays etc.

### Default Argument Resolution

Given
```
double GetAverage<typename T>(const std::vector<T>& someNumbers);
```
then
```
std::vector<double> myNumbers;
double result = GetAverage(myNumbers);
```
will call
```
double GetAverage<double>(const std::vector<double>& someNumbers);
```
So, if function parameters can inform the compiler uniquely as to which function to instantiate, its automatically compiled. 

### Explicit Argument Resolution

However, given
```
double GetAverage<typename T>(const std::vector<T>& someNumbers);
```
and
```
std::vector<int> myIntegers;
double result = GetAverage(myIntegers);
```
But you don't want the int version called, you can
```
double result = GetAverage<double>(myIntegers);
```
i.e. name the template parameter explicitly.

### Code Bloat
Given (a rather stupid example)
```
double GetMax<typename T1, typename T2>(const &T1, const &T2);
```
and
```
double r1 = GetMax(1,2);
double r2 = GetMax(1,2.0);
double r3 = GetMax(1.0,2.0);
```
The compiler will generate 3 different max functions.

* Be Careful
    * Executables/libraries get larger
    * Compilation time will increase
    * Error messages get more verbose
    
### Two Stage Compilation

* Basic syntax checking (eg. brackets, semi-colon, etc)
* Further checks when function is instantiated (eg. existence of + operator).

### Instantiation

* Template functions can be
    * .h file only
    * .h file that includes .cxx/.txx file (e.g. ITK)
    * .h file and separate .cxx/.txx file
* Object Code is only really generated if code is used
* If
    * Compilation Unit A defines function
    * Compilation Unit B uses function
    * What about 3 scenarios above?
* In general
    * Most prefer header only implementations
    
### Explicit Instantiation
See also [this](http://en.cppreference.com/w/cpp/language/function_template).

* The using class (B) explicitly instantiates the function at the end of file.
* Instantiation must only occur once (use Header Guard)


### Implicit Instantiation

* Instantiated as they are used
* via ```#include```



