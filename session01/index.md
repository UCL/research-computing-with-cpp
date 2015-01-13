---
title: C++ Recap
---

## Introduction

### Assumed Knowledge 

On MPHYGB24 you learnt:

* Lecture 4: Compiling a library, testing debugging
* Lecture 5: Arrays
* Lecture 6: Structures, dynamically allocated arrays
* Lecture 7: Classes
* Lecture 8: Operator overloads, inheritance
* Lecture 9: Polymorphism

You should have equivalent knowledge as a pre-requisite.


### Aim for Today

* Provide a reminder of 
    * C++ concepts ([MPHYGB24][MPHYGB24])
    * CMake usage ([MPHYGB24][MPHYGB24])
* Unit Testing concepts
* Unit Testing in practice
  
From the start of this course we encourage Test/Behaviour driven development.

Scientific software should be as rigorous as sterile lab techniques.
  

## Basic C++ features

### Classes
 
* Procedural programming: pass data to functions
    * Can get out of hand as program size increases
    * Can't easily describe relationships between bits of data
    * Can't easily control access to data
* Object oriented programming: describe types and how they interact


### Abstraction

* Enables you to define a type
    * Class defines concept or "blueprint"
    * We "instantiate" to create a specific object 
* Example: Fraction data type
{{cppfrag('01','fraction/fraction.h')}}


### Encapsulation

* Encapsulation is:
    * Bundling together methods and data
    * Restricting access, defining public interface
* For class methods/variables:
    * `private`: only available in this class
    * `protected`: available in this class and derived classes
    * `public`: available to anyone with access to the object
    
    
### Inheritance

* Used for:
    * Defining new types based on a common type
    * Reduce code duplication, less maintenance
* Careful:
    * Types in a hierarchy MUST be related
    * Don't over-use this, just to save code duplication
    * There are other ways 
* Example: Shapes
{{cppfrag('01','shape/shape.h')}}


### Polymorphism

* Several types:
    * "subtype": via inheritance
    * "parametric": via templates
    * "ad hoc": via function overloading
* In C++, normally we refer to "subtype" polymorphism
* Is the provision of a common interface to entities of different types
* Example: Shape
{{cppfrag('01','shape/shapeTest.cc')}}


### Further Reading

* Every C++ developer should keep reading
    * [Effective C++][Meyers], Meyers
    * [More Effective C++][Meyers], Meyers
    * [Effective STL][Meyers], Meyers
    * Design Patterns (1994), Gamma, Help, Johnson and Vlassides


## Various Tips

### Practical Tips

* If you feel like:
    * More coding, more things go wrong
    * Everything gets messy
    * Feeling that you're digging a hole
* Then we provide:
    * Pragmatic tips as how to do this in practice
    * In a scientific research sense


### Coding tips

* Follow coding conventions for your project 
* Compile often
* Version control
    * Commit often
    * Useful commit messages - don't state what can be diff'd, explain why.
    * Short running branches
    * Covered on [MPHYG001][MPHYG001]    
* Class: "does exactly what it says on the tin"
* Class: build once, build properly, so testing is key.


### C++ tips

Numbers in brackets refer to Scott Meyers "Effective C++" book.

* Declare data members private (22)
* Initialise objects properly. Throw exceptions from constructors. (4) 
* Use `const` whenever possible (3) 
* Make interfaces easy to use correctly and hard to use incorrectly (18) 
* Prefer non-member non-friend functions to member functions (better encapsulation) (23) 
* Avoid returning "handles" to object internals (28) 
* Never throw exceptions from destructors


### OO tips

* Make sure public inheritance really models "is-a" (32) 
* Learn alternatives to polymorphism (Template Method, Strategy) (35) 
* Model "has-a" through composition (38) 
* Understand [Dependency Injection][DependencyInjection].
* i.e. most people overuse inheritance


### Scientific Computing tips

* Papers require numerical results, graphs, figures, concepts
* Optimise late
    * Correctly identify tools to use
    * Implement your algorithm of choice
    * Provide flexible design, so you can adapt it and manage it
    * Only optimise the bits that are slowing down the production of interesting results
* So, this course will provide you with an array of tools


## CMake

### CMake Introduction

* This is a practical course
* We need to run code
* Use CMake as a build tool
* CMake produces
    * Windows: Visual Studio project files
    * Linux: Make files
    * Mac: XCode projects, Make files
* This course will provide CMake code and boiler plate code


### CMake Usage

Typically, to do an "out-of-source" build

```
cd ~/myprojects
git clone http://github.com/somecode
mkdir somecode-build
cd somecode-build
cmake ../somecode
make
```

    
## Unit Testing

### What is Unit Testing?

At a high level

* Way of testing code. 
* Unit
    * Smallest 'atomic' chunk of code
    * i.e. Function, Class
* See also:
    * Integration Testing
    * System Testing
    * User Acceptance Testing
    
    
### Benefits of Unit Testing?

* Certainty of correctness
* Influences and improves design
* Confidence to refactor, improve
* Continuous improvement, development


### Drawbacks for Unit Testing?

* Takes too much time
    * Really?
* Don't know how
    * This course will help
* IT WILL SAVE TIME in the long run


### Unit Testing Frameworks

Generally, very similar

* JUnit (Java), NUnit, CppUnit, phpUnit, 
* Basically
    * Macros (C++), methods (Java) to test conditions
    * Macros (C++), reflection (Java) to run/discover tests
    * Ways of looking at results.
        * Java/Eclipse: Integrated with IDE
        * Log file or standard output


## Unit Testing Example
        
### How To Start 

We discuss

* Basic Example
* Some tips

Then its down to the developer/artist.


### C++ Frameworks

To Consider:

* [Catch][Catch]
* [GoogleTest][GoogleTest]
* [QTestLib][QTestLib]
* [BoostTest][BoostTest]
* [CppTest][CppTest]
* [CppUnit][CppUnit]


### Worked Example

* Borrowed from
    * [Catch Tutorial][CatchTutorial]
    * and [Googletest Primer][GoogleTestPrimer]
* We use [Catch], so notes are compilable.
* But the concepts are the same


### Code

To keep it simple for now we do this in one file:

{{cppfrag('01','factorial/factorial1.cc')}}

Produces this output when run:

{{execute('01','factorial/factorial1')}}


### Principles

So, typically we have

* Some `#include` to get test framework
* Our code that we want to test
* Then make some assertions


### Catch / GoogleTest

For example, in [Catch][Catch]:

```
    // TEST_CASE(<unique test name>, <test case name>)
    TEST_CASE( "Factorials are computed", "[factorial]" ) {
        REQUIRE( Factorial(2) == 2 );
        REQUIRE( Factorial(3) == 6 );
    }
```

In [GoogleTest][GoogleTest]:

```
    // TEST(<test case name>, <unique test name>)
    TEST(FactorialTest, HandlesPositiveInput) {
      EXPECT_EQ(2, Factorial(2));
      EXPECT_EQ(6, Factorial(3));
    }
```

all done via C++ macros.


### Tests That Fail

What about Factorial of zero?
Adding
 
 ```
    REQUIRE( Factorial(0) == 1 );
```

Produces something like:

```
    factorial2.cc:9: FAILED:
    REQUIRE( Factorial(0) == 1 )
    with expansion:
    0 == 1
```

### Fix the Failing Test

Leading to:

{{cppfrag('01','factorial/factorial2.cc')}}

Which passes:

{{execute('01','factorial/factorial2')}}


### Test Macros

Each framework has a variety of macros to test for failure. [Check][Check] has:

```
    REQUIRE(expression); // stop if fail
    CHECK(expression);   // doesn't stop if fails
```

if an exception is throw, its caught, reported and counts as a failure.

Examples:

```
    CHECK( str == "string value" );
    CHECK( thisReturnsTrue() );
    REQUIRE( i == 42 );
```

Others:

```
    REQUIRE_FALSE( expression )
    CHECK_FALSE( expression )
    REQUIRE_THROWS( expression ) # Must throw an exception
    CHECK_THROWS( expression ) # Must throw an exception, and continue testing
    REQUIRE_THROWS_AS( expression, exception type )
    CHECK_THROWS_AS( expression, exception type )
    REQUIRE_NOTHROW( expression )
    CHECK_NOTHROW( expression )
```    
    
### Testing for Failure
    
To re-iterate:
    
* You should test failure cases
    * Force a failure
    * Check that exception is thrown
    * If exception is thrown, test passes
    * (Some people get confused, expecting test to fail)
* Examples
    * Saving to invalid file name
    * Negative numbers passed into double arguments
    * Invalid Physical quantities (e.g.  -300 Kelvin)

    
### Setup/Tear down
  
* Some tests require objects to exist in memory
* These should be set up
    * For each test
    * for a group of tests
* Frameworks do differ in this regards

    
### Example
    
Referring to the [Catch Tutorial][CatchTutorial]:
 
``` 
TEST_CASE( "vectors can be sized and resized", "[vector]" ) {

    std::vector<int> v( 5 );

    REQUIRE( v.size() == 5 );
    REQUIRE( v.capacity() >= 5 );

    SECTION( "resizing bigger changes size and capacity" ) {
        v.resize( 10 );

        REQUIRE( v.size() == 10 );
        REQUIRE( v.capacity() >= 10 );
    }
    SECTION( "resizing smaller changes size but not capacity" ) {
        v.resize( 0 );

        REQUIRE( v.size() == 0 );
        REQUIRE( v.capacity() >= 5 );
    }
    SECTION( "reserving bigger changes capacity but not size" ) {
        v.reserve( 10 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 10 );
    }
    SECTION( "reserving smaller does not change size or capacity" ) {
        v.reserve( 0 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 5 );
    }
}

```
 
So, Setup/Tear down is done before/after each section.    
    
    
## Quick Tips

### C++ design

* Stuff from above applies to Classes / Functions
* Think about arguments:
    * Code should be hard to use incorrectly.
    * Use `const`, `unsigned` etc.
    * Testing forces you to sort these out.


### BDD vs TDD

* Test Driven Development
    * Test/Design based on methods available
* Behaviour Driven Development
    * Test/Design based on behaviour
    
subtly different.


### Anti-Pattern 1: Setters/Getters

Testing every Setter/Getter. 

Consider:

```
   class Atom {
     
     public:
       void SetAtomicNumber(const int& number) { m_AtomicNumber = number; }
       int GetAtomicNumber() const { return m_AtomicNumber; }
       void SetName(const std::string& name) { m_Name = name; }
       std::string GetName() const { return m_Name; }
     private:
       int m_AtomicNumber;
       std::string m_Name;
   };
```

and tests like:

```
    TEST_CASE( "Testing Setters/Getters", "[Atom]" ) {
    
        Atom a;
    
        a.SetAtomicNumber(1);
        REQUIRE( a.GetAtomicNumber() == 1);
        a.SetName("Hydrogen");
        REQUIRE( a.GetName() == "Hydrogen");
```

* It feels tedious.
* But you want good coverage.
* This often puts people off testing.
* It also produces "brittle", where 1 change brakes many things.


### Anti-Pattern 1: Suggestion.

* Focus on behaviour.
    * What would end-user expect to see. 
    * How would end-user be using this class.
    * Write tests that follow the use-case.
    * Gives a more logical grouping.
    * One test can cover > 1 function.
    * i.e. move away from slavishly testing each function.
* Minimise interface.
    * Provide the bare number of methods.
    * Don't provide setters if you dont want them.
    * Don't provide getters unless the user needs something.
    * Less to test. Use documentation to describe why.
    
        
### Anti-Pattern 2: Constructing Dependent Classes

* Sometimes, by necessity we test groups of classes.
* Or one class genuinely Has-A contained class.
* But the contained class is expensive, or could be changed in future


### Anti-Pattern 2: Suggestion

* Read up on [Dependency Injection][DependencyInjection]
* Enables you to create and inject dummy test classes
* So, testing again used to break down design, and increase flexibility.


### Summary BDD Vs TDD

Aim to write:

* Most concise description of requirements as unit tests.
* Smallest amount of code to pass tests.
* ... i.e. based on behaviour


## The End

### Any questions?

[MPHYGB24]: https://moodle.ucl.ac.uk/course/view.php?id=5395
[Meyers]: http://www.aristeia.com/books.html
[MPHYG001]: https://moodle.ucl.ac.uk/course/view.php?id=28759
[DependencyInjection]: http://en.wikipedia.org/wiki/Dependency_injection
[GoogleTest]: https://code.google.com/p/googletest/
[QTestLib]: http://qt-project.org/doc/qt-4.8/qtestlib-manual.html
[CppUnit]: http://sourceforge.net/projects/cppunit/
[CppTest]: http://cpptest.sourceforge.net/
[Catch]: https://github.com/philsquared/Catch
[BoostTest]: http://www.boost.org/doc/libs/1_57_0/libs/test/doc/html/index.html
[GoogleTestPrimer]: https://code.google.com/p/googletest/wiki/V1_7_Primer
[CatchTutorial]: https://github.com/philsquared/Catch/blob/master/docs/tutorial.md
[DependencyInjection]: http://martinfowler.com/articles/injection.html