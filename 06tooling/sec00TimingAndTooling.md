---
title: Timing and Tooling
---

This week we'll look a bit at how to time our code for performance, and also introduce a number of tools which we can use to develop and improve our code. We'll start using these in the practical but please make sure you install and set up the tools ahead of time, and reach out to us before class if you have trouble doing so. 

## Timing 

Timing statments can be inserted into the code usind the `<chrono>` header from the standard library. 

`std::chrono` allows us to time code using a number of different clocks:
- `system_clock`: This is the system wide real time clock.
    - Be aware that system time can be adjusted while your program is running (e.g. by admin) which would interfere with measurements.
- `steady_clock`: A monotonically increasing clock which cannot be adjusted. Good for measuring time differences, but won’t give you the time of day when something happens.
- `high_resolution_clock`: The clock with the shortest ’tick’ time available on the system - this may be system_clock or steady_clock or neither, so the properties of this clock are system dependent. A useful choice when your primary concern is precision (e.g. timing functions which are fairly short), as long as you can be confident that your clock won't be alterned during the run. 

We can see an example of how to do timing using the following code fragment:

```cpp
#include <iostream>
#include <chrono>

typedef std::chrono::steady_clock timingClock;

void my_function()
{
    // your interesting code here
}

int main()
{
    std::chrono::time_point<timingClock> t_start = timingClock::now();
    my_function();
    std::chrono::time_point<timingClock> t_end = timingClock::now();

    std::chrono::nanoseconds diff = t_end - t_start;

    std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(diff);

    double seconds = static_cast<double>(duration.count()) * 1e-6;

    std::cout << "Time taken = " << seconds << std::endl;
}
```
- We can take the time at a given point using the `now()` function on a particular clock. 
- Take these times as close on either side of the thing that you want to measure as possible. Don't put additional code (especially slow things like I/O) inside your timing statments unless you want them to contribute to yours times! 
- `now()` returns a `time_point` type. The difference of two `time_points` is a duration type, which we can cast between different units such as `nanoseconds` and `microseconds`. 
- We can convert a duration type to a numerical using `count()`, which by default is an integral type. This can be cast to a floating point type such as `double` if you want to use the time with fractional artihmetic. 

Some things to note about this code:
- As you can see, the types in `chrono` are quite verbose due to the amount of namespaces!
- I very strongly recommend using `typedef` or `using` statments to reduce clutter (the code would have been _even longer_ if we haven't created the type alias `timingClock`). 
- I've written out the types of everything explicitly here so that you can see what types each of these functions returns, but in practice (once you're familiar with the way that `chrono` works) this can be a good place to use `auto` to de-clutter. 
- This code will work for any kind of clock, so you can change the clock you are using by simply changing the `typedef` statement at the top and leaving the rest of the code unchanged. 


A more succinct version of this code might look like:
```cpp
#include <iostream>
#include <chrono>

typedef std::chrono::steady_clock timingClock;

void my_function()
{
    // Your interesting code here
}

int main()
{
    auto t_start = timingClock::now();
    my_function();
    auto t_end = timingClock::now();

    std::chrono::nanoseconds diff = t_end - t_start;

    double seconds = static_cast<double>(diff.count()) * 1e-9;

    std::cout << "Time taken = " << seconds << std::endl;
}
```
- `auto` saves writing and reading complicated types for a function which is obvious.
- It's not obvious that a difference in two timepoints will default to nanoseconds, so that type should (in my opinion) be kept in the code for clarity. This is particularly true since it's necessary to see that the calculation of `seconds` is correct. 
    - We can also use other types. `std::chrono::nanoseconds` is actually an alias for the type `std::chrono::duration<int64_t, std::nano>`. The first template parameter is the type that you want your `count` to go into (which can be integral or floating point types), and the second is essentially our units (`std::nano`, `std::micro`, `std::milli` etc.). You can use your own template instantiation `std::chrono::duration<double, std::nano>` if you want to skip the `static_cast`. 
    - You can use `std::nano` in combination with integral types (`int`, `int64_t`, `size_t`, `uint` etc.) or floating point types (`float`, `double` etc.). You can use `std::micro` with floating point types but if you want to have an integral count representing the number of microseconds then you need to a duration in nanoseconds first and then do a `duration_cast`. 
    - **Since there are so many options here, it's a good idea to just tell people what type you're using!**
- We showed above that we _can_ convert to microseconds and so on, but we don't have to do so, we can work directly with `nanoseconds` if that's useful.
- It can also be a good idea to just wrap up some timing code in a little class so you can reuse it across projects and don't have to keep thinking about all this stuff. 

## Tooling

**N.B.** Please remember that if you are using Windows for this course you will need to install these tools **inside WSL** (Windows Subsystem for Linux) rather than following a normal Windows installation. To do so, you can 
1. Open a Command Prompt and type `wsl` to go into the WSL command line. From there you can follow Linux instructions for installing command line tools like Valgrind. 
2. Open VSCode and [connect to WSL using the button in the bottom left hand corner](https://code.visualstudio.com/docs/remote/wsl). From there you can add extensions to VSCode, or open a terminal to access the WSL command line and install command line tools. 

## Debugging inside VSCode

We can debug our code from inside VSCode but it requires a little setup to make sure we're correctly using CMake when debugging. Follow [this tutorial to set up your VSCode properly with CMake](https://code.visualstudio.com/docs/cpp/CMake-linux).

## Debugging memory issues with Valgrind

If you're unlucky enough to have to resort to unsafe memory management with raw pointers, you will almost certainly meet a **segmentation fault** or segfault, if your program tries to access memory it doesn't strictly have access to. This can happen due to many different types of bugs; stack overflows, freeing already freed pointers, off-by-one bugs in loops, etc, but can be notoriously tricky to debug.

Valgrind is a **memory profiler and debugger** which can do many useful things involving memory but we just want to introduce its ability to find and diagnose segfaults by tracking memory allocations, deallocations and accesses.

You should follow [Valgrind's Quickstart Guide](https://valgrind.org/docs/manual/quick-start.html).

## Linting with clang-tidy

**Linters** are tools that statically analyse code to find common bugs or unsafe practices. We'll be playing with the linter from the Clang toolset, `clang-tidy` so follow this tutorial on setting up clang-tidy with VSCode:

{% include youtube_embed.html id="8RSxQ8sluG0" %}  

## Formatting with clang-format

If you've done much Python programming you probably already know the power of good formatters, tools that reformat your code to a specification. This can help standardise code style across codebases and avoid horrid debates about spaces vs tabs, where curly braces should go, or how many new lines should separate functions.

Again, we'll be peeking into the Clang toolbox and using `clang-format` to automatically format our code. Follow [this guide on setting up a basic .clang-format file](https://leimao.github.io/blog/Clang-Format-Quick-Tutorial/) and see clang-format's [list of common style guides](https://clang.llvm.org/docs/ClangFormatStyleOptions.html#basedonstyle) for more information about what styles are available. Look at a few, choose one you like and use that style to format your assignment code.

You can also use [the clang-format VSCode extension](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format) to automatically format your code on saving.

## Compiler warnings

One of the easiest ways to improve your code is to turn on **compiler warnings** and fix each warning. Some companies even require that all compiler warnings are fixed before allowing code to be put into production. Check out [this blog post on Managing Compiler Warnings with CMake](https://www.foonathan.net/2018/10/cmake-warnings/) for details on how to do this in our CMake projects. I recommend you use these warnings to fix potential bugs in your assignment.

## Optional: Performance profiling with gprof

Although you won't be required to use one on this course, as we move towards *performant* C++, one useful tool is a **profiler**. This is a tool that runs your code and measures the time taken in each function. This can be a powerful way to understand which parts of your code need optimising. 

There are many advanced profilers out there but a good, simple profiler is `gprof`. This also has the advantage of coming with most Linux distributions, so is automatically available with Ubuntu on either a native Linux machine or WSL. 

You can watch this introductory video on using gprof:

{% include youtube_embed.html id="zbTtVW64R_I" %}  

and try profiling one of your own codes. Since we're using cmake, we can't directly add the required `-pg` flags to the compiler so we'll have to tell cmake to add those flags with:

```
cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg ...
```

On MacOS you can try using Google's [gperftools](https://github.com/gperftools/gperftools) which is available through homebrew.

- You should target the areas of your code where your application spends the most time for optimisation. 
- Profilers are excellent for identifying general behaviour and bottlenecks, but you may be able to get more accurate results for specific functions or code fragments by inserting timing code. 

