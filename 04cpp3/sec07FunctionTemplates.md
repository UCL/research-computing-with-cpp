---
title: Function Templates
---

## Function Templates

### Function Templates Example

* Credit to [www.cplusplus.com][OverloadedFunctions]

{% idio cpp/sumFunctionExample %}

{% code sumFunctionExample.cc  %}

* And produces this output when run

{% code sumFunctionExample.out %}

{% endidio %}


### Why Use Function Templates?

* Instead of function overloading
    * Reduce your code duplication
    * Reduce your maintenance
    * Reduce your effort
    * Also see this [Additional tutorial][TemplatesTutorial].


### Language Definition 1

* From the [language reference](http://en.cppreference.com/w/cpp/language/function_template)


```
template < parameter-list > function-declaration
```

* so

```
template < class T >  // note 'class'
void MyFunction(T a, T b)
{
  // do something
}
```

* or

```
template < typename T1, typename T2 >  // note 'typename'
T1 MyFunctionTwoArgs(T1 a, T2 b)
{
  // do something
}
```


### Language Definition 2

* Also
    * Can use ```class``` or ```typename```.
    * I prefer ```typename```.
    * Template parameter can apply to references, pointers, return types, arrays etc.


### Default Argument Resolution

* Given:

```
double GetAverage<typename T>(const std::vector<T>& someNumbers);
```

* then:

```
std::vector<double> myNumbers;
double result = GetAverage(myNumbers);
```

* will call:

```
double GetAverage<double>(const std::vector<double>& someNumbers);
```

* So, if function parameters can inform the compiler uniquely as to which function to instantiate, its automatically compiled.


### Explicit Argument Resolution - part 1

* However, given:

```
double GetAverage<typename T>(const T& a, const T& b);
```

* and:

```
int a, b;
int result = GetAverage(a, b);
```

* But you don't want the int version called (due to integer division perhaps), you can:

```
double result = GetAverage<double>(a, b);
```


### Explicit Argument Resolution - part 2

* equivalent to 

```GetAverage<double>(static_cast<double>(a), static_cast<double>(b));```

* i.e. name the template function parameter explicitly.

* Cases for Explicit Template Argument Specification
    * Force compilation of a specific version (eg. int as above)
    * Also if method parameters do not allow compiler to deduce anything eg. ```PrintSize()``` method.


### Beware of Code Bloat

* Given:

```
double GetMax<typename T1, typename T2>(const &T1, const &T2);
```

* and:

```
double r1 = GetMax(1,2);
double r2 = GetMax(1,2.0);
double r3 = GetMax(1.0,2.0);
```

* The compiler will generate 3 different max functions.
* Be Careful
    * Executables/libraries get larger
    * Compilation time will increase
    * Error messages get more verbose


### Two Stage Compilation

* Basic syntax checking (eg. brackets, semi-colon, etc), when ```#include```'d
* But only compiled when instantiated (eg. check existence of + operator).


### Instantiation

* Object Code is only really generated if code is used
* Template functions can be
    * .h file only
    * .h file that includes separate .cxx/.txx/.hxx file (e.g. ITK)
    * .h file and separate .cxx/.txx file (sometimes by convention a .hpp file)
* In general
    * Most libraries/people prefer header only implementations


### Explicit Instantiation - part 1

* Language Reference [here][FunctionTemplate]
* [Microsoft Example][ExplicitInstantiationMicrosoft]

{% idio cpp/explicitInstantiation %}

* Given (library) header:

{% code explicitInstantiation.h %}

* Given (library) implementation:

{% code explicitInstantiation.cc %}


### Explicit Instantiation - part 2

* Given client code:

{% code explicitInstantiationMain.cc %}

* We get:

{% code explicitInstantiationMain.out %}

{% endidio %}


### Explicit Instantiation - part 3

* Explicit Instantiation:
    * Forces instantiation of the function
    * Must appear after the definition
    * Must appear only once for given argument list
    * Stops implicit instantiation
* So, mainly used by compiled library providers
* Clients then ```#include``` header and link to library

```
Linking CXX executable explicitInstantiationMain.x
Undefined symbols for architecture x86_64:
  "void f<float>(float)", referenced from:
```


### Implicit Instantiation - part 1

* Instantiated as they are used
* Normally via ```#include``` header files.

{% idio cpp/implicitInstantiation %}

* Given (library) header, that containts implementation:

{% code implicitInstantiation.h %}


### Implicit Instantiation - part 2

* Given client code:

{% code implicitInstantiation.cc %}

* We get:

{% code implicitInstantiation.out %}

{% endidio %}

### Homework 19

* Write a template function `AGreaterThanB` that compares two input agruments of type `T` and returns a `bool` if `A` is greater than `B`. The function should be able to handle either `int`,  `float` or `string` entries (for the latter you will need to decide how to rank by size) 
* Try out the different types of explicit and implicit instantiation 
* Advanced/optional: write a template function that performs a binary search on the contents of a vector containing either `int`,  `float` or `string` entries (you could adapt this [equivalent to](http://www.cplusplus.com/reference/algorithm/find/) ) code, N.B. the `vector` will need to be sorted by size

[OverloadedFunctions]: http://www.cplusplus.com/doc/tutorial/functions2 'Overloaded Functions and Template Functions'
[FunctionTemplate]: http://en.cppreference.com/w/cpp/language/function_template 'Function Template Reference'
[TemplatesTutorial]: http://www.codeproject.com/Articles/257589/An-Idiots-Guide-to-Cplusplus-Templates-Part 'Templates Tutorial'
[ExplicitInstantiationDisc]: http://stackoverflow.com/questions/2351148/explicit-instantiation-when-is-it-used 'Explicit Instantiation Discussion'
[ExplicitInstantiationMicrosoft]: http://msdn.microsoft.com/en-us/library/by56e477%28VS.80%29.aspx 'Microsoft Explicit Instantiation Example'
