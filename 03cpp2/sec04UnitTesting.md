---
title: Testing Software
---

# Testing Software

Testing is a crucial part of of software projects, and is especially critical in projects like scientific software where it is necessary to be confident about the accuracy and correctness of our results. 

Testing is usually a layered process, with:
- **unit tests** focussing on checking small, independent pieces of code, and
- **integration tests** checking larger processes that involve multiple pieces of unit tested code working together, and
- **system tests** checking that the complete system works.

## Testing Frameworks

Testing frameworks exist for all major languages, and you may have come across some of them before. In this course we will be using the [Catch2](https://github.com/catchorg/Catch2) library for writing tests. If, in the future, you decide (or need) to use a different framework such a [Google Tests](https://github.com/google/googletest), then the process will be the same but with some changes in syntax. The same approach can also be taken to other languages like Python using frameworks like [pytest](https://docs.pytest.org/en/7.4.x/), which some of you may have used before. 

## Installing Catch2

**Before this week's class you need to install the Catch2 testing library.**

You can clone the [Catch2 repository here](https://github.com/catchorg/Catch2). To install you should complete the following steps:

1. Clone the repository. 
2. Move into the Catch2 folder in your terminal. 
3. `cmake -B build -DBUILD_TESTING=OFF`
4. `cmake --build build`
5. If you have permissions and want to install system wide you can run `cmake --install build/`. Otherwise run `cmake --install build/ --prefix install_path` where you replace `install_path` a path to a folder of your choice. 

To make Catch2 available in CMake, in your top level CMakeLists.txt include the following line: 

`find_package(Catch2 3 REQUIRED)`

or if you have not installed system wide, use the following line:

`find_package(Catch2 3 REQUIRED PATHS install_path)`

replacing `install_path` with the path to your chosen install folder. This allows cmake to find your Catch2 installation, including the header files and the compiled library. 

When you create your executable for your test files, you'll need to link with the Catch2 library (along with any other libraries you need):
```cmake
add_executable(test_executable)
target_sources(test_executable PRIVATE test_source.cpp)
target_link_libraries(test_executable PUBLIC Catch2::Catch2WithMain)
```

You can find more information about using Catch2 in the documentation on their [github page](https://github.com/catchorg/Catch2).

## Unit Testing Principles

Unit tests check the correctness of the smallest pieces of code, for example an individual function or class with no further dependencies. Such a function may have multiple _test cases_, which check for different aspects of the behaviour, or which check the behaviour under different circumstances/inputs. 

When writing unit tests for a function you will want to consider:
- What are some where you _know_ the expected output? It is important not to write circular tests which end up comparing the result of the code to itself!
- You should always test the output on multiple inputs. 
    - Make sure that you don't only check trivial cases. Often special cases can be easier to calculate the expected values for, but don't require all the code to be correct to get the answer you expect. For example settings values in your input vector to 0, or passing an empty vector, might get you the right result even if your code processing that vector is wrong. Ask yourself if the outputs that you're checking are dependent on all the different parts of the code you are testing being correct. If not, then add more test cases that do depend on those bits of code!
- Break your set of possible inputs into different cases, and pay particular attention to edge cases.
    - You may want to check for example how your function behaves for positive numbers, negative numbers, and 0. For some functions you may want to check how it handles very large or small inputs (if there is a risk of overflow/underflow). 
    - Edge cases are the most common places for errors to appear. Consider a function that takes a vector of data and updates it so that each value is replaced by the average of two values on either side of it. What should happen to the values at either end of the vector? Does this happen correctly? 
- Checking for failure is as important as checking for success! Many functions that we write are not valid for all inputs, and we should check that they behave properly when given invalid inputs. 
    - Check that functions which should throw exceptions under certain circumstances do indeed throw those exceptions. 
    - Common failure cases for various kinds of functions include providing negative or 0-value inputs, indices which are out of bounds, or empty vectors/strings. 
    - If you have designed a class to have certain properties, check that you cannot construct an invalid class, and that its properties are maintained correctly. 
- Ask yourself if another function, which does something different to what you intended, could pass your tests. If so, consider making your tests more robust! 

Writing comprehensive tests takes some practice, and is highly dependent on the kind of code that you have written. **Always look at your code critically, consider how it behaves in different circumstances.** 

## Integration and System Tests

Once you have thoroughly tested the smallest units of your code, you should also write tests which test more complex components. These may be functions which call multiple sub-functions for example. You should increase the size of the components you are testing piece by piece until you reach full system tests. System tests check the behaviour from input to output, and often include mocking up things like user input. 

## Testing Syntax 

Most of the important syntax that you will need to know about are contained within two header files:
```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
```
- `catch_test_macros.hpp` contains the macros that we need in order to declare test cases and do standard checks. 
- `catch_matchers_floating_point.hpp` contains some extra functionality for testing floating point values. This is different from testing other values because **floating point arithmetic is almost never completely accurate, so instead we have to test whether results are within some relative or absolute tolerance.** 
    - There is additional special functionality for other things like strings, containers (such as vector) and more. `<catch2/matchers/catch_matchers_all.hpp>` can be useful if you want to have access to all of these. 
    - I recommend writing the line `using namespace Catch::Matchers` if you are using any of this functionality to avoid namespace clutter in your test files. (Otherwise you need to write things like `Catch::Matchers::WithinRel` every time you check a float which gets very cumbersome and makes things more difficult to read.)

Here's an example using our `vector_functions` code from last week's class, with some additional example tests to check floating point and vector examples. 

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include "vector_functions.h"
#include <functional>

using namespace Catch::Matchers;

TEST_CASE( "Counting with loop is correct", "[ex1]" ) {
    std::vector<int> example = {5};
    REQUIRE( countMultiplesOfFive(example) == 1);
}

TEST_CASE( "Adding elements to a vector", "[ex2]" )
{
    std::vector<int> starting_vector;
    addElements(starting_vector, 5, 4);
    REQUIRE( starting_vector.size() == 4);
    for(auto &&i : starting_vector)
    {
        REQUIRE(i == 5);
    }
}

TEST_CASE("Testing floats", "[matchers]")
{
    double x = 1.0/2.0;
    REQUIRE_THAT(x, WithinRel(0.5, 0.0001));
}

TEST_CASE("Testing vectors", "[matchers]")
{
    std::vector<int> v{1, 2, 3, 4, 5};
    std::vector<bool> v_check(v.size());
    std::transform(v.begin(), v.end(), v_check.begin(), [](int z){return z > 0;});
    REQUIRE_THAT(v_check, AllTrue());
}
```
- `TEST_CASE` declares a new test case within this test file. Each test case has a description and optional tags. The tags can be used to run subsets of tests. Tests can have more than one tag, and the same tag can be used for multiple tests. 
- `REQUIRE` checks that the statement within the brackets is true.
- `REQUIRE_THAT` takes a value first, and then a matcher expression. Matcher expressions do something more complex than just a simple equality test. 
    - `WithinRel` is a matcher expression which checks that takes two params: the expected value and the relative tolerance. 
    - `AllTrue` is a matcher expression that takes no arguments, and is used to check that every element in an iterable container is true. 

### REQUIRE and CHECK

When using the `REQUIRE` macro, a `TEST_CASE` will terminate if it fails, and execution will move on to the next `TEST_CASE`. When using the `CHECK` macro, the `TEST_CASE` will continue on whether it passes or fails. 

- If your `TEST_CASE` does independent tests which don't require the previous checks to have passed in order to continue, then use `CHECK`.
- If you need a particular condition to pass before continuing -- a good example is checking an object being correctly set up -- then use `REQUIRE`. 
- Similarly we have both `REQUIRE_THAT` and `CHECK_THAT` for matchers. 

### Testing Floating Point

You can use `WithinAbs(value, tolerance)` and `WithinRel(value, tolerance)` to check that floating point numbers are within some reasonable tolerance of what you expect. When considering what tolerance to use, you should think about:
- What is the precision of your floating point numbers? (`float` or `double`?)
- How much error might accumulate in your calculation?
- How much error is acceptable to you? What precision is actually required?  

### Testing for Exceptions

You should test that exceptions are thrown when they should be, and in some cases you explicitly check that exceptions are not thrown when they shouldn't be. 
- Test any cases where exceptions should be thrown. 
    - Use `REQUIRE_THROWS_AS(expression, exception_type)` or `CHECK_THROWS_AS(expression, exception_type)` to check that the correct exception is thrown. For example, if we have a factorial function which throws a `logic_error` if the factorial is undefined (e.g. negative numbers) then you could write `CHECK_THROWS_AS(factorial(-1), std::logic_error)`. 
- If a constructor can throw an exception, you should check that it does not throw an exception for a valid instantiation using `CHECK_NOTHROW` or `REQUIRE_NOTHROW`. 

**You can consult the Catch2 documentation for even more macros and ways of testing your code.**