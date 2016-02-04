---
title: Template Meta-Programming
---

## Template Meta-Programming (TMP)

### What Is It?

* See [Wikipedia][TMPWikipedia], [Wikibooks][TMPWikiBooks], [Keith Schwarz][TMPKeithSchwarz]
* C++ Template
    * Type or function, parameterised over, set of types, constants or functions
    * Instantiated at compile time
* Meta Programme
    * Program that produces or manipulates constructs of target language
    * Typically, it generates code
* Template Meta-Programme
    * C++ programme, uses Templates, generate C++ code at compile time


### TMP is Turing Complete

* Given: A [Turing Machine][TuringMachine]
    * Tape, head, states, program, etc.
* A language is "Turing Complete" if it can simulate a Turing Machine
    * e.g. Conditional branching, infinite looping
* Turing's work underpins much of "what can be computed" on a modern computer
    * C, C++ no templates, C++ with templates, C++ TMP
    * All Turing Complete
* Interesting that compiler can generate such theoretically powerful code.    
* But when, where, why, how to use TMP?    
* (side-note: Its not just a C++ pre-processor macro)    


### Why Use It?

* Use sparingly as code difficult to follow
* Use for
    * Optimisations
    * Represent Behaviour as a Type
    * Traits classes
* But when you see it, you need to understand it!


### Factorial Example

See [Wikipedia Factorial Example][TMPWikipedia]

{% idio cpp/TMPFactorial %}

* This:

{% code TMPFactorial.cc %}

* Produces:

{% code TMPFactorial.out %}

{% endidio %}

### Factorial Notes:

* Compiler must know values at compile time
    * i.e. constant literal or constant expression
    * See also [constexpr][C++11constexpr]
* Generates/Instantiates all functions recursively
* Factorial 16 = 2004189184
* Factorial 17 overflows
* This simple example to illustrate "computation"
* But when is TMP actually useful?
* Notice that parameter was an integer value ... not just "int" type


### Loop Example

{% idio cpp/TMPLoopUnrolling %}

* This:

{% code TMPLoop.cc %}

* Time: numberOfInts=3 took 40 seconds


### Loop Unrolled

* This:

{% code TMPLoopUnrolled.cc %}

{% endidio %}

* Time: numberOfInts=3 took 32 seconds when switch to fixed vector, and 23 when a raw array.  


### Policy Checking

* Templates parameterised by type not by behaviour
* But you can make a class to represent the behaviour
* See [Keith Schwarz][TMPKeithSchwarz] for longer example.


### Simple Policy Checking Example

{% idio cpp/TMPPolicy %}

* This:

{% code TMPPolicy.cc %}

* Produces:

{% code TMPPolicy.out %}

{% endidio %}

### Summary of Policy Checking Example

* Define interface for behaviour
* Parameterize over all behaviours
* Use multiple-inheritance to import policies
* e.g. logging / asserts


### Traits

* From C++ standard 17.1.18
    * "a class that encapsulates a set of types and functions necessary for template classes and template functions to manipulate objects of types for which they are instantiated."
* Basically: Traits represent details about a type
* You may be using them already!
* Start with a simple example


### Simple Traits Example

{% idio cpp/TMPTrait %}

* This:

{% code TMPTrait.cc %}

* Produces:

{% code TMPTrait.out %}

{% endidio %}

### Traits Principles

* Small, simple, normally public, eg. struct
* else/if
    * Else template
    * partial specialisations
    * full specialisations
* Probably using them already
    * ```std::numeric_limits<double>::max()```
    * ITK has similar ```itk::NumericTrait<PixelType>```
* Applies to primatives as well as types    


### Traits Examples

* [Simple Tutorial from Aaron Ballman][AaronBallman]
* [Boost meta-programming support][BoostMetaProg]
* [Boost type_traits tutorial][BoostTutorial]
* [C++11 has many traits][C++11traits]


### Wait, Inheritance Vs Traits?

* We said inheritance is often overused in OO
* We say that too frequent if/switch statements based on type are bad in OO
* C++11 providing many [is_X type traits][C++11traits] returning bool, leading to if/else
* So, when to use it?


### When to use Traits

* Some advice
    * Sparingly
    * To add information to templated types
    * Get algorithm to work for 1 data type
    * If you extend to multiple data types and consider templates
        * When you need type specific behaviour
            * traits probably better than template specialisation
            * traits better than inheritance based template hierarchies

* Remember
    * Scientist = few use-cases
    * Library designer = coding for the unknown, and potentially limitless use-cases
        * More likely of interest to library designers

### TMP Use in Medical Imaging - 1

Declare an [ITK][ITK] image

``` cpp

template< typename TPixel, unsigned int VImageDimension = 2 >
class Image:public ImageBase< VImageDimension >
{
public:
// etc

```

* TPixel, ```int```, ```float``` etc.
* VImageDimension = number of dimensions


### TMP Use in Medical Imaging - 2

But what type is origin/spacing/dimensions?

``` cpp
template< unsigned int VImageDimension = 2 >
class ImageBase:public DataObject
{
  typedef SpacePrecisionType                          SpacingValueType;
  typedef Vector< SpacingValueType, VImageDimension > SpacingType;
```


### TMP Use in Medical Imaging - 3

So now look at ```Vector```

``` cpp
template< typename T, unsigned int NVectorDimension = 3 >
class Vector:public FixedArray< T, NVectorDimension >
{
public:
```


### TMP Use in Medical Imaging - 4

Now we can see how fixed length arrays are used

``` cpp
template< typename T, unsigned int TVectorDimension >
const typename Vector< T, TVectorDimension >::Self &
Vector< T, TVectorDimension >
::operator+=(const Self & vec)
{
  for ( unsigned int i = 0; i < TVectorDimension; i++ )
    {
    ( *this )[i] += vec[i];
    }
  return *this;
}
```

which may be unrolled by compiler.


### TMP Use in Medical Imaging - 5

* [ITK][ITK]
    * uses [```itk::NumericTraits<>```][ITKNumericTraits] adding mathematical operators like multiplicative identity, additive identity
    * uses traits to describe features of meshes, ```like numeric_limits```, but more generalised
* [MITK][MITK] (requires coffee and a quiet room)
    * uses [mitkPixelTypeList.h][MITKPixelType] for multi-plexing across templated image to non-templated image type
    * uses [mitkGetClassHierarchy.h][MITKClassHierarchy] to extract a list of class names in the inheritance hierarchy
* [TMP in B-spline based registration][BSplinePaper]:


### Further Reading For Traits

* [Keith Schwarz][TMPKeithSchwarz]
* [Nathan Meyers][NathanMeyers]
* [Todd Veldhuizen, traits scientific computing][TraitsScientificComputing]
* [Thaddaaeus Frogley, ACCU, traits tutorial][ThaddaaeusFrogley]
* [Aaron Ballman][AaronBallman]
* [Andrei Alexandrescu][AndreiAlexandrescuTraits]
* [Andrei Alexandrescu traits with state][AndreiAlexandrescuTraitsWithState]
* [Boost meta-mrogramming support][BoostMetaProg]
* [Boost type_traits tutorial][BoostTutorial]
* [C++11 has many traits][C++11traits]


### Further Reading In General

* [Andrei Alexandrescu's Book][AndreiAlexandrescuBook]
* [Herb Sutter][Sutter]'s [Guru of The Week][GOTW], especially [71][GOTW71] and [this][GOTWInheritanceVsTraits] article
* And of course, keep reading [Meyers][Meyers]


### Summary

* Learnt
    * Notation for template function/class/meta-programming
    * Uses and limitations of template function/class
    * Template Meta-Programming
        * Optimisation, loop unrolling
        * Policy classes
        * Traits


[TMPWikipedia]: http://en.wikipedia.org/wiki/Template_metaprogramming
[TMPWikibooks]: http://en.wikibooks.org/wiki/C%2B%2B_Programming/Templates/Template_Meta-Programming
[TMPKeithSchwarz]: http://www.keithschwarz.com/talks/slides/tmp-cs242.pdf
[TuringMachine]: http://en.wikipedia.org/wiki/Turing_machine
[TuringComplete]: http://en.wikipedia.org/wiki/Turing_completeness
[C++11constexpr]: http://en.wikipedia.org/wiki/C%2B%2B11#constexpr_.E2.80.93_Generalized_constant_expressions
[C++11traits]: http://www.cplusplus.com/reference/type_traits
[BoostMetaProg]: http://www.boost.org/doc/libs/?view=category_Metaprogramming
[BoostTutorial]: http://www.boost.org/doc/libs/1_57_0/libs/type_traits/doc/html/boost_typetraits/background.html
[ThaddaaeusFrogley]: http://accu.org/index.php/journals/442
[TraitsScientificComputing]: http://www.cs.rpi.edu/~musser/design/blitz/traits.html
[NathanMeyers]: http://www.cantrip.org/traits.html
[AaronBallman]: http://blog.aaronballman.com/2011/11/a-simple-introduction-to-type-traits
[AndreiAlexandrescuTraits]: http://erdani.com/publications/traits.html
[AndreiAlexandrescuTraitsWithState]: http://erdani.com/publications/traits_on_steroids.html
[AndreiAlexandrescuBook]: http://www.amazon.co.uk/Modern-Design-Generic-Programming-Patterns/dp/0201704315/ref=sr_1_1?ie=UTF8&qid=1421739179&sr=8-1&keywords=andrei+alexandrescu
[GOTW]: http://www.gotw.ca/gotw
[GOTW71]: http://www.gotw.ca/gotw/071.htm
[GOTWInheritanceVsTraits]: http://www.gotw.ca/publications/mxc++-item-4.htm
[Sutter]: http://herbsutter.com
[ITKNumericTraits]: http://www.itk.org/Doxygen/html/classitk_1_1NumericTraits.html
[MITK]: http://www.mitk.org
[MITKPixelType]: http://docs.mitk.org/2014.03/mitkPixelTypeList_8h.html
[MITKClassHierarchy]: http://docs.mitk.org/nightly-qt4/mitkGetClassHierarchy_8h.html
[BSplinePaper]: http://link.springer.com/chapter/10.1007%2F978-3-319-08554-8_2
[Meyers]: http://www.aristeia.com/books.html
[ITK]: http://www.itk.org

