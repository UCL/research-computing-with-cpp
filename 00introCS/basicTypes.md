---
title: Basic Types
---

# Basic Types in Programming Languages

In order to effectively work in any programming language, we need a basic understanding of a number of fundamental types. We will give these types by the names in C++, although there are generally analogues in other languages. 

## Boolean Types

Boolean types represent true/false values. In C++ the type is called `bool` and it takes 1 byte of memory.

## Integer Types

All programming languages will have some types to describe integer values. We use these to represent things that in mathematics are in the set of _natural numbers_ $\mathbb{N}$ or _integers_ $\mathbb{Z}$. In C++ we might most commonly use:

- `int`: integers that can be positive or negative, and take 4 bytes of memory.
- `unsigned int`: integers that can only be positive, and take 4 bytes of memory. Because they don't have to represent negative numbers as well, `unsigned int` can represent larger values than `int`. 
- `std::size_t`: unsigned integers that can represent that size of the largest possible object; this is usually determined by the addressing on an architecture. For example, on a 64-bit system `size_t` will typically be 8 bytes. This is the standard type to use for things like indexing arrays/vectors since it can't be negative and is guaranteed to be able to represent the largest index of an element of an array. 

There are many other types that C++ offers, although not all of them are standard. You should use the C++ documentation to check whether a type that you want to use is standardised and portable across different architectures. (As an example, I used to use `uint` for unsigned integers until some Mac users informed me that it doesn't compile with the `clang` compiler!)

## Strings and Characters

A single character can be represented using `char`, which takes a single byte. For strings you are advised to use the C++ type `std::string`. In some places you will come across `char*`, an C-style array of `char`, as an alternative to C++ strings. These should only be used where necessary for compatibility. The most common place for this to come up is command line arguments for the `main` function.

## Floating Point Representation of Real Numbers

The real numbers, $\mathbb{R}$, is the set of numbers along the number line. There are infinitely many real numbers in any finite interval, and this poses a problem for computation: we clearly cannot represent all the real numbers exactly, since we have a finite amount of space in which to represent them. Floating point representation is how we typically deal with approximating the real numbers in code. Floating point numbers can in principle have any level of precision, but the most common are:

- `float`: 32 bits (4 bytes),
- `double`: 64 bits (8 bytes),
- `long double`: Usually 80 or 128 bits depending on processor,
- `half`: 16 bits, not part of the C/C++ standard but often used on hardware like GPUs.

The data for floating point numbers are split into two parts, the mantissa and the exponent. It is comparable to scientific notation (when you write a number as $a \times 10^b$) except that it uses powers of 2 instead of 10. We can write a floating point number $n$ as 

$n = m \times 2^e$

where $m$ is the mantissa and $e$ is the exponent. The mantissa is usually represented such that there is a radix point after the first significant (non zero) digit. In other words, the binary `001011` would represent the mantissa `1.011` (again, in base 2). The exponent is a signed integer. 

The size of the datatype affects both the precision of the mantissa and the range of values for the exponent (and therefore how large and small the values represented can be). 

- `float` has around 7 significant figures
- `double` has around 16 significant figures

Floating point computation **is not exact**. 

- Adding values of very different sizes leads to significant loss of precision since values must be converted to have the same exponent to be added together. This means the difference in scale is pushed into the mantissa, which then loses precision due to leading `0` digits on the smaller number. In some cases the smaller number may be so small that the closest representable number with that exponent is `0` and so the addition is lost completely. 
- Subtracting values which are close in size leads to cancellation of many digits and a result with far fewer significant digits and therefore lower precision. 
- Identities from real arithmetic do not necessarily hold, in particular **addition and multiplication are not associative**, so $(a + b) + c \neq a + (b + c)$ in floating point! 
- Handling these difficulties in numerical methods is a major field in and of itself. Many numerical algorithms are specially crafted to correct rounding errors in floating point arithmetic.  

### Floating Point Precision

Higher precision floating point numbers (like `double` as opposed to `float`) will give more accurate results when doing numerical work, but may also be slower to perform operations. Historically `double` operations have taken more time to compute than `float` operations, although this is no longer typically the case on modern CPUs. Nevertheless, if you are exploiting SIMD registers for maximal performance, fewer `double` values can fit in an available register and therefore fewer operations can be performed in a given amount of time. Some fast algorithms use single precision `float` or even half precision floating point numbers in areas where the results will not be significantly impacted by this. This is particularly common in areas like statistics and machine learning where the statistical variance is much larger than the precision of the floating point numbers. You must always bear in mind that the use of lower precision floating point types can lead to numerical instability from cancellation errors or division/multiplication by extreme values causing under or overflow. You should always have tests for robustness and precision to check that any compromises made to precision are acceptable. 

## Containers

Containers are important to most programming languages; almost all programmers will need to work on things like lists of data, or vectors and matrices. Some commonly used containers, that we will use repeatedly throughout this course, are:

- `std::array<T, N>`: a type for storing exactly `N` elements of type `T`, with `N` known at compile time.
- `std::vector<T>`: a type for storing arbitrary numbers of elements of type `T`.
- `std::map<A, B>`: a type for storing key-value pairs where keys are unique and have type `A` and values have type `B`.

