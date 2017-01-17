---
title: Unit Testing
---

## Unit Testing

### What is Unit Testing?

At a high level

* Way of testing code.
* Unit
    * Smallest 'atomic' chunk of code
    * i.e. Function, could be a Class
* See also:
    * Integration/System Testing
    * Regression Testing
    * User Acceptance Testing


### Benefits of Unit Testing?

* Certainty of correctness
* (Scientific Rigour)
* Influences and improves design
* Confidence to refactor, improve


### Drawbacks for Unit Testing?

* Don't know how
    * This course will help
* Takes too much time
    * Really?
    * IT SAVES TIME in the long run


### Unit Testing Frameworks

Generally, all very similar

* JUnit (Java), NUnit (.net?), CppUnit, phpUnit,
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
* We use [Catch], so notes are compilable
* But the concepts are the same


### Code

To keep it simple for now we do this in one file:

{% idio cpp/factorial %}

{% code factorial1.cc %}

Produces this output when run:

{% code factorial1.out %}


### Principles

So, typically we have

* Some `#include` to get test framework
* Our code that we want to test
* Then make some assertions


### Catch / GoogleTest

For example, in [Catch][Catch]:

``` cpp
    // TEST_CASE(<unique test name>, <test case name>)
    TEST_CASE( "Factorials are computed", "[factorial]" ) {
        REQUIRE( Factorial(2) == 2 );
        REQUIRE( Factorial(3) == 6 );
    }
```

In [GoogleTest][GoogleTest]:

``` cpp
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

``` cpp
    REQUIRE( Factorial(0) == 1 );
```

Produces something like:

``` cpp
    factorial2.cc:9: FAILED:
    REQUIRE( Factorial(0) == 1 )
    with expansion:
    0 == 1
```

### Fix the Failing Test

Leading to:

{% code factorial2.cc %}

which passes:

{% code factorial2.out %}

{% endidio %}


### Test Macros

Each framework has a variety of macros to test for failure. [Check][Check] has:

``` cpp
    REQUIRE(expression); // stop if fail
    CHECK(expression);   // doesn't stop if fails
```

if an exception is throw, its caught, reported and counts as a failure.

Examples:

``` cpp
    CHECK( str == "string value" );
    CHECK( thisReturnsTrue() );
    REQUIRE( i == 42 );
```

Others:

``` cpp
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


### Setup/Tear Down

* Some tests require objects to exist in memory
* These should be set up
    * for each test
    * for a group of tests
* Frameworks do differ in this regards


### Setup/Tear Down in Catch

Referring to the [Catch Tutorial][CatchTutorial]:

``` cpp
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


## Unit Testing Tips

### C++ design

* Stuff from above applies to Classes / Functions
* Think about arguments:
    * Code should be hard to use incorrectly.
    * Use `const`, `unsigned` etc.
    * Testing forces you to sort these out.


### Test Driven Development (TDD)

* Methodology
    1. Write a test
    1. Run test, should fail
    1. Implement/Debug functionality
    1. Run test
        1. if succeed goto 5
        1. else goto  3
    1. Refactor to tidy up


### TDD in practice

* Aim to get good coverage
* Some people quote 70% or more
* What are the downsides?
* Don't write 'brittle' tests


### Behaviour Driven Development (BDD)

* Behaviour Driven Development (BDD)
    * Refers to a [whole area](https://en.wikipedia.org/wiki/Behavior-driven_development) of software engineering
    * With associated tools and practices
    * Think about end-user perspective
    * Think about the desired behaviour not the implementation
    * See [Jani Hartikainen](https://codeutopia.net/blog/2015/03/01/unit-testing-tdd-and-bdd/) article.

### TDD Vs BDD

* TDD
    * Test/Design based on methods available
    * Often ties in implementation details
* BDD
    * Test/Design based on behaviour  
    * Code to interfaces (later in course)
* Subtly different
* Aim for BDD


### Anti-Pattern 1: Setters/Getters

Testing every Setter/Getter.

Consider:

``` cpp
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

``` cpp
    TEST_CASE( "Testing Setters/Getters", "[Atom]" ) {

        Atom a;

        a.SetAtomicNumber(1);
        REQUIRE( a.GetAtomicNumber() == 1);
        a.SetName("Hydrogen");
        REQUIRE( a.GetName() == "Hydrogen");
```

* It feels tedious
* But you want good coverage
* This often puts people off testing
* It also produces "brittle", where 1 change breaks many things


### Anti-Pattern 1: Suggestion.

* Focus on behaviour.
    * What would end-user expect to be doing?
    * How would end-user be using this class?
    * Write tests that follow the use-case
    * Gives a more logical grouping
    * One test can cover > 1 function
    * i.e. move away from slavishly testing each function
* Minimise interface.
    * Provide the bare number of methods
    * Don't provide setters if you don't want them
    * Don't provide getters unless the user needs something
    * Less to test. Use documentation to describe why.


### Anti-Pattern 2: Constructing Dependent Classes

* Sometimes, by necessity we test groups of classes
* Or one class genuinely Has-A contained class
* But the contained class is expensive, or could be changed in future


### Anti-Pattern 2: Suggestion

* Read up on [Dependency Injection][DependencyInjection]
* Enables you to create and inject dummy test classes
* So, testing again used to break down design, and increase flexibility


### Summary BDD Vs TDD

Aim to write:

* Most concise description of requirements as unit tests
* Smallest amount of code to pass tests
* ... i.e. based on behaviour


## Any Questions?

### End of Lecture?

* Example git repo, CMake, Catch template project:
    * ```[https://github.com/MattClarkson/CMakeCatchTemplate]```(https://github.com/MattClarkson/CMakeCatchTemplate)
    
[GoogleTest]: https://code.google.com/p/googletest/
[QTestLib]: http://qt-project.org/doc/qt-4.8/qtestlib-manual.html
[CppUnit]: http://sourceforge.net/projects/cppunit/
[CppTest]: http://cpptest.sourceforge.net/
[Catch]: https://github.com/philsquared/Catch
[BoostTest]: http://www.boost.org/doc/libs/1_57_0/libs/test/doc/html/index.html
[GoogleTestPrimer]: https://code.google.com/p/googletest/wiki/V1_7_Primer
[CatchTutorial]: https://github.com/philsquared/Catch/blob/master/docs/tutorial.md
