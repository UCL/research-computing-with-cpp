---
title: Timing
---

# Timing 

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


