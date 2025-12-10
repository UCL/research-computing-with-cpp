---
title: Variadic Templates
---

# Variadic Templates and Fold Expressions

Up to now, we have only worked with templates that take a fixed number of parameters. C++17 introduces two features that make it possible to write functions which work with _any number_ of arguments: variadic template parameter packs and fold expressions.

As with all template code, the compiler must see the full template definition wherever it is instantiated, so **functions using variadic templates and fold expressions are normally placed in header files**.

## Parameter Packs

A _parameter pack_ allows a function template to take an arbitrary number of arguments of the same logical category. For example, we might want a function that computes an operation over several numbers. We will look at two examples which use common patterns for writing variadic template functions:
```cpp
template<typename T, typename... Ts>
T variadic_sum(Ts... xs);

template <typename T, typename... Ts>
T quadrature(T x, Ts... xs);
```
You may  wonder why the latter template is written with both a single type `typename T` and a parameter pack `typename... Ts` in the arguments, rather than putting everything into one pack. The structure is intentional and it helps in two important ways:
- By separating out the first argument and using it as the return type it is clearer what type will be returned by a call and what implicit conversions may take place if the function is called with a mix of types (e.g. `double`, `float`, `int`), rather than leaving it up to the compiler (which may also fail to infer the output type if there is not sufficient information). Every subsequent argument in the paramter pack must be convertible to the same type `T`. 
- In general, **a parameter pack may be empty**. For some functions like finding the maximum of a list however the operation only makes sense if there is at least one value to operate on. By writing the function as:
    ```cpp
    T quadrature(T x, Ts... xs)
    ```
    the first value `x` must always be present. If we instead wrote:
    ```cpp
    template <typename... Ts>
    auto quadrature (Ts... xs);
    ```
    then the coompiler would happily accept call like `quadrature` (zero arguments), which is not well defined. Splitting off the first argument avoids this situation without needing additional error-handling constructs.

### Unpacking Parameter Packs

There are a variety of ways of extracting information from the parameter packs. One very important piece of information is the _number of arguments_ in the pack. We can extract this using the `sizeof...()` operator. (Note that the elipsis `...` is part of the function name in this case!)

```cpp
template<typename T, typename... Ts>
T variadic_sum(T x, Ts... args)
{
    const int N = sizeof...(args);

}
```

This function is of course incomplete! One thing that we can do is convert a pack to an array of variables:

```cpp
template<typename T, typename... Ts>
T variadic_sum(Ts... args)
{
    const int N = sizeof...(args);
    T ys[N] = {args...};  // get array of arguments
    
    T sum = 0;
    for(int i = 0; i < N; i++)
    {
        sum += ys[i];
    }

    return sum;
}
```

This gives us our array of arguments, but note that we have now imposed a restriction: since `ys` is an array of `T`, all of the arguments must be implicitly converted to type `T` here. 

We can then call the function as follows:
```cpp
int main()
{
    double s = variadic_sum<double>(1.0, 2.0, 4, 5.1);
    printf("Sum = %f\n", s);

    return 0;
}
```
Note that we have had to specify the first type (`T`) as the compiler is unable to infer it. You can specify as many of the parameters as you need to, **in order**. For example, the following are all valid function calls:

```cpp
    double s1 = variadic_sum<double>(1.0, 2.0, 4, 5.1);
    double s2 = variadic_sum<double, double, double, double>(1.0, 2.0, 4, 5.1);
    double s2 = variadic_sum<double, double, int>(1.0, 2.0, 4, 5.1);
```
In the definition of `s2` the `4` is implicitly converted into a `double` before being passed into the function. since that function takes the first four arguments as `double`. In the definition of `s1` it is passed as an `int` and converted when we define `ys`, since this is an array of `T`, which is given type `double`. In the definition of `s3` the `4` is again passed as an `int` and is converted when `ys` is defined.

We can also write a range based loop to avoid potential sizing errors:

```cpp
template<typename T, typename... Ts>
T variadic_sum(Ts... args)
{
    T sum = 0;
    for(const auto& y : {args...})
    {
        sum += y;
    }

    return sum;
}
```

A special consideration is needed here because we have not explicitly converted the `args...` to an array of a specific type. The compiler will try to do this to `{args...}` automatically, but if not all of our arguments are the same type then will be a compilation error. As such, our mixed `double` and `int` arguments will fail here, and we would need to make sure all arguments are explicitly doubles like so:

```cpp
int main()
{
    double s1 = variadic_sum<double, double, double, double>(1.0, 2.0, 4, 5.1);
    double s2 = variadic_sum<double>(1.0, 2.0, 4.0, 5.1);  // 4.0 is interpreted as double
}
```

## Fold Expressions

A more concise way of writing functions over parameter packs can often be found by using C++17 fold expressions. A fold is a kind of _reduction_ expression. A fold takes a list of elements $(x_0, x_1, ..., x_{N-1})$, a binary operator $\oplus$, and a special element $i$ and applies a binary operator $\oplus$ to each of the elements. For an associative operator, a reduction looks like this:

$i \oplus x_0 \oplus x_1 \oplus ... \oplus x_{N-1}$.

However, if the operator is non-associative, that is if $x \oplus (y \oplus z) \neq (x \oplus y) \oplus z$, then we can distinguish between right folds

$(x_0 \oplus (x_1 \oplus (... \oplus(x_{N-1} \oplus i) ...)))$,

and left folds

$(...((i \oplus x_0) \oplus x_1) \oplus x_2) \oplus ... ) \oplus x_{N-1}$,

by their order of operations and by whether the special element $i$ appears is applied to the first or last element. ($i$ is usually the _identity_ element of $\oplus$, so it usually does not matter whether this element goes, and the concern between right and left folds is generally the order of operations for non-associative operators.)

Fold expressions allow us to apply an operator across all elements of a parameter pack in a single, compact expression. This is particularly useful for numerical operations like sums and products. 

Let us examine the most simple case of our `variadic_sum` again:

```cpp
template<typename T, typename... Ts>
T variadic_sum(Ts... args)
{
    T sum = (0 + ... + args);  // left fold expression

    return sum;
}
```
The fold expression `(0 + ... + args)` automatically expands depending on how many arguments are provided. This avoids writing separate overlaods for 2, 3 or more inputs, and significantly reduces boilerplate.

Note that we have started fold with `0`, which is the _identity_ element of the addition operator ($x + 0 = x$ for all $x$). This is very common in reduction expressions, for example in products you will often start with `1` which is the multiplicative identity. You can, of course, start with any element you like, for example `T sum = (5 + ... + args);` would give $5 + \Sigma_i y_i$. 

We can also write this as a right fold expression:

```cpp
template<typename T, typename... Ts>
T variadic_sum(Ts... args)
{
    T sum = (args + ... + 0);  // right fold expression

    return sum;
}
```

Mathematical addition is associative, but remember that on a computer _floating point addition is not perfectly associative due to rounding errors_, and therefore your results may vary depending on your order of summation (i.e. left vs right fold). 

There is a [list of operators available for use in fold expressions](https://en.cppreference.com/w/cpp/language/fold.html), however we can also make our expressions can be more complex than a simple reductions; as an example in scientific computing it is common to combine independent uncertainties ($\sigma_i$) by adding them in quadrature:
$$
\sigma_\mathrm{tot} = \sqrt{\sigma_0^2 + \sigma_1^2 + \ldots + \sigma_{N-1}^2}.
$$
A variadic template with a fold expression provides a natural, compact implementation:
```cpp
#include <cmath>

template <typename T, typename... Ts>
constexpr T quadrature(T x, Ts... xs) {
  const T sumsq = (x*x + ... + (xs*xs)); // left fold expression 
  return std::sqrt(sumsq);
}
```
Note that by separating off the first argument we are forcing this function to take at least one argument, and can therefore also dispense with any speical element $i$ like the identity. 

Here the binary operator is again addition but we are squaring the elements before they are added. You can use folds to calculate expressions of the form

$(f(x_0) \oplus (f(x_1) \oplus (... \oplus(f(x_{N-1} \oplus i)) ...)))$,

or

$(...(((i \oplus f(x_0)) \oplus f(x_1)) \oplus f(x_2)) \oplus ... ) \oplus f(x_{N-1})$

for some function $f$, or more generally you can replace $f(x)$ with an _expression_ which depends on $x$. For example `(std::cout << x)` is an expression but not a function; the following variadic function prints out whatever arguments are supplied.

```cpp
void print_args(Ts... args)
{
    ((std::cout << args << " "), ..., (std::cout << std::endl));
}
```

In this case ["`,`" is the operator](https://en.cppreference.com/w/cpp/language/operator_other.html#Built-in_comma_operator), which just evaluates expressions separated by a comma left-to-right, and `std::cout << args << " "` is the expression that is applied to each variable, and `std::cout << std::endl` is the special element (which is also an expression because `,` operates on expressions). 

### "Unary" Fold Expressions

We can also write left and right fold expressions as a so-called "unary fold" expression. This does _not_ mean that the operator is a unary operator: $\oplus$ is always a binary operator! Instead it means that we don't supply the special element $i$, like this:

```cpp
template<typename T, typename... Ts>
T unary_sum(Ts... args)
{
    T sum = (args + ...);  // Unary right fold

    return sum;
}
```

Because there is no special element $i$ defined, this function is undefined for an empty parameter pack, and therefore will not compile if called with no arguments. This is another way of enforcing a non-empty argument list without needing to separate out the first argument. 


