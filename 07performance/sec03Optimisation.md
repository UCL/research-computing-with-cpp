---
title: Compiler Optimisation
---

Estimated Reading Time: 45 minutes

# Compiler Optimisation and Compile-Time Evaluation

Compilation is the translation of our high level code (in this case C++) into machine code that reflects the instruction set of the specific hardware for which it is compiled. This machine code can closely reflect the C++ code, implementing everything explicitly the way that it's written, or it can be quite different from the structure and form of the C++ code **as long as it produces an equivalent program***. The purpose of this restructuring is to provide optimisations, usually for speed. Modern compilers have a vast array of optimisations which can be applied to code as it is compiled to the extent that few people could write better optimised assembly code manually, a task that rapidly becomes infeasible and forbiddingly time consuming as projects become larger and more complex. 

There is another benefit to automated compiler optimisation. Compilers, by necessity, do produce hardware specific output, as they must translate programs into the instruction set of a given processor. This means that even if we have written highly portable code which makes no hardware specific optimisations, we can still benefit from these optimisations if they can be done by the compiler when compiling our code for different targets! As we shall see below, some processors may have different features such as machine level instructions for vectorised arithmetic which can be implemented by the compiler without changing the C++ code, producing different optimised programs for different hardware from a single, generic C++ code.   

As such, to get the best out of our C++ code we need to rely to some extent on automated optimisation by the compiler. This does not mean that we should not choose effective algorithms -- the compiler will not simply replace a slow sorting algorithm with a better one! -- but rather compiler optimisation should be used in conjunction with our own best practices for writing efficient software. 

> *_What is considered an "equivalent program" is beyond the scope of this course, and falls under the field of programming language semantics (the study of the meaning of programs). For now you can take equivalence to mean that the results of two computations are the same whenever they provided with the same external inputs. Things like the CPU clock, and therefore timing information, would qualify as external inputs, and so changing the timing results of a computation doesn't change the meaning of the program._

## Optimisation Trade Offs

Code with optimisations applied will generally run faster, but there are a number of other impacts that it can also have that are worth bearing in mind when selecting appropriate optimisations to apply. 

1. Executable Size. 
    - The size of the compiled code can be increased by optimisation. Although one might expect code which is optimised to also be simplified and smaller, there are many optimisations which increase the size of the resulting machine code. An example of this would be _loop unrolling_. (See below)
2. Debugging Experience.
    - One of the most useful tools for debugging code is the ability to step through a program and check the values in variables as you execute line-by-line. Optimised code can make many changes, removing redundant variables, changing branching logic, restructuring or removing loops etc., that affect the correspondence between the C++ code and the compiled machine code. As a result, it may not always be possible to step through an optimised code, or be meaningful to ask about the value of a specific variable at a particular point in the execution of a code. 
3. Compilation Time.
    - Optimised compilation involves making complex transformations to your code, and depending on the nature and size of your code and the optimisations applicable, this may take a long time. 
4. Standards Compliance.
    - Some optimisations are not compliant with floating point standards; in particular they may affect floating point computations by rearranging numerical expressions ("free re-associations"). Using these can jeopardise the accuracy of your programs. 

## Constant Expressions and Compile-Time Evaluation

We can tell the compiler to do some computations at compile-time instead of during run-time. Consider some simple code like this:

```cpp
int x = 5;
int y = x*x + 12;
int z = factorial(5);
```

The variables $y$ and $z$ are the result of simple, deterministic expressions that depend only on information that we have at compile time. An equivalent program could look like this:

```cpp
int x = 5;
int y = 37;
int z = 120;
```

This program clearly doesn't need to do any work at run-time in order to assign values to `y` and `z`, but we have lost the expressiveness of our original version (which makes the relationships between `x`, `y`, and `z` clear), and we would have to update `y` and `z` manually if we changed the initialisation of `x`. 

**Constant expressions** provide us with a way of ensuring that we can write expressive code like the first example _and_ forcing the compiler to evaluate the expressions and replace them with their results at compile time. This is in particular useful for more complex functions that could be time-consuming and runtime and are less likely to be automatically optimised by the compiler. 

### Constant Expression Syntax

A variable or a function can be declared as a constant expression using `constexpr`. For example:

```cpp
constexpr int add(int a, int b)
{
    int c = a + b;
    return c;
}

int main()
{
    constexpr int x = 5;                  
    constexpr int y = add(x, 18);

    return 0;
}
```

The initialization of `x` is essentially equivalent to `const x = 5` in this case. There is an additional restriction on `constexpr` compared to `const` however, which is that the initialisation of a `const` variable can happen at runtime, and depend on the run-time state, whereas the initialisation of a `constexpr` variable must be able to be performed at compile time.  

The use of `constexpr` for the function `add` enforces that this function can be evaluated on compile time and will not rely on any runtime information. This means that the variable assignment for `y` does not require runtime calculation, but the compiler will simplify `add(x, 18)` to `23` and simply assign the value to `y` without ever calling the function when the program is run.

### Limitations of Constant Expressions

Not all computations can be done at compile-time, and therefore there are a number of conditions that constant expressions must fulfil. The [complete list of conditions](https://en.cppreference.com/w/cpp/language/constexpr.html) is quite involved, but here are some key points to remember:

- A `constexpr` function cannot contain `try` statements, since handling exceptions would require run-time state. 
    - You can _write_ a `throw` statement in a `constexpr` function but if the `throw` statement is reached during the compile time evaluation you will get a compiler error. This is useful because it will warn you if there is an error case in a compile-time evaluation, and because you can also call `constexpr` functions at run-time as well which can then handle the exception. 
- A `constexpr` function cannot include uninitialised variables (e.g. `int z;`).
- A `constexpr` function cannot declare `static` variables, since they require maintaining a state between function calls. 
- A `constexpr` function cannot call a non-`constexpr` function. This includes things like dynamic memory allocation with `new`. However since a `constexpr` function can call other `constexpr` functions, you can also write a _recursive_ `constexpr` function!
- A `constexpr` function cannot declare a _non-literal_ type variable. [Literal types](https://en.cppreference.com/w/cpp/language/constant_expression.html#Literal_type) need to fulfil a variety of conditions, in essence they are simple types that can be worked with at compile time, so they must be able to be constructed using a `constexpr`, have trivial destructors (no custom destructor logic needs to be called), and not contain member variables of non-literal types. An example of a non-literal type is `std::vector` since it has non-trivial destruction logic (heap memory deallocation), but `std::array` _is_ a literal type since its size is known at compile time and so it can be stack-allocated and doesn't require any specialised destruction logic.
- A `constexpr` variable must be a literal type, and must be initialised by a constant expression e.g. a call to a `constexpr` function with argument _known at compile time_ (e.g. an explicit number like `2.4` or a `const`/`constexpr` variable known at compile-time).

If your function is a _pure function_ that doesn't involve exception handling, dynamic memory allocation/deallocation, or calls to non-`constexpr` functions, then you can probably turn it into a `constexpr` function. 

**N.B. We are using C++17 on this course. The conditions for acceptable constant expressions vary across different C++ standards, so if you use a newer or older standard you may find some differences in what you are able to write and compile.**

### Revisiting the Factorial Example

Consider the following `constexpr` definition of `factorial`, where we have omitted `#include` statements and `std::` namespacing for brevity.

```cpp
constexpr int factorial(int x)
{
    if(x < 0)
    {
        throw domain_error("Value " + to_string(x) + " is not within the domain of factorial (x >= 0).");
    }
    else if(x == 0)
    {
        return 1;
    }
    else
    {
        return x*factorial(x-1);
    }
}
```
Notice that we are able to make use of branching logic with separate `return` statements and recursion in this definition.

Now consider the following statements that could appear in main, and whether or not they will compile:

```cpp
int main()
{
    const int x = 5;                  
    int y = 3;

    constexpr int z = factorial(x);   // Compiles okay: x is const
    constexpr int r = factorial(y);   // Doesn't compile: y is not const
    constexpr int p = factorial(-3);  // Doesn't compile: throw is reached
    int q = factorial(-3);            // Compiles okay: throws an exception at runtime

    return 0;
}
```
Notice that we can only initialise a `constexpr int` variable with a call to `factorial` if the **argument is `const` and the function evaluation does not `throw`**. If we initialise a non-`const` variable using `factorial` then it will be treated as a runtime expression and so these conditions don't apply. 

### Other Uses of Constant Expressions

Constant expressions are not just useful for run-time optimisation; they also allow use to write more expressive code anywhere where we need to know information at compile time, such as constant template arguments (e.g. the length of an `std::array`) and static memory allocation (e.g. `int x[...];`).

## Compiler Optimisation Flags

The GNU compiler (gcc) has a [large number of optimisation options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html), but for the most part one uses a smaller set of flags which enable batches of these options. These batches are selected to give you control over some of the downsides of optimisation procedures discussed above.
 
- No flag is equivalent to `-O0`, which means that the optimisations are turned off. This results in machine code that most closely matches your C++ code as you have written it. 
- `-O1` performs a basic set of optimisations for speed, skipping optimisations which tend to inflate compilation times significantly. 
- `-O2` performs all the optimisations from `-O1` but also additional optimisations which may affect compile times more drastically, but which do not have a speed-space trade off (i.e. avoids inflating the size of the executable too much). 
- `-O3` performs all the `-O2` optimisations and some additional ones which may impact the size of the executable. It does not turn on any optimisations which are not standards compliant. 
- `-Og` performs some optimisations but keeps compile times short and strictly disables various optimisations that make certain structural changes to the code. This is designed for debugging so that you can step through your code line by line. 
- `-Ofast` performs aggressive optimisation on numerical code but sacrifices compliance with some standards, meaning that the result of your calculation can be affected. 

There are others in the documentation linked above which I would recommend that you read. It also describes in detail the optimisations that are turned on at each level. 

## Some Optimisation Examples

In the following sections we will look at a few common compiler optimisations and their implications. It's important to bear in mind however that these optimisations are not _guaranteed_ to happen, and optimisation is usually a heuristic process. Analysing programs is complex, and you are certainly not guaranteed to get the optimal version of your compiled code out of the compiler, and the compiler will make decisions about what optimisations to apply where based on various rules of thumb that are different for each compiler (and target architecture). If you _need_ an optimisation to be applied that isn't guaranteed by the C++ standard then you should consider implementing it yourself directly in your source code. 

### Compile-time Calculations and Redundancy

If the compiler is given an expression which it can calculate and replace with a value at compile time, then it may do so without changing the meaning of the program. As a simple example:

```c++
const int x = 12;
const int y = 8;

int z = x + y;
```

- The compiler knows the values of `x` and `y`, and they can't have been changed (partly because there is no intervening code, and also because they are declared `const`). So at compile time the compiler can deduce that the initial value of `z` will be `20`. The compiler can replace the addition operation with a hard coded initialisation value.

The compiler may also notice that certain variables are not accessed or used, or that the result of a calculation is thrown away, and therefore that some calculation can be avoided. In general it's good practice to remove any redundant variables or calculations yourself! (Remember that you can turn on compiler warnings to help with this.) 

You may find that this kind of optimisation can negatively affect simple benchmarks that you write. Let's say we want to benchmark a sorting algorithm:

```c++
vector<int> unsorted = gen_list(1000'000);
auto t_start = std::chrono::high_resolution_clock::now();
vector<int> sorted = sort(unsorted); 
auto t_end = std::chrono::high_resolution_clock::now();
```

If our program never accesses `sorted` before it terminates (because we were only interested in the timing information), and the compiler can tell that `sort` does not have side-effects, then the compiler may recognise that this calculation is redundant and skip it entirely! When benchmarking with optimisations on you may have to force the program to actually do the work you're interested in, e.g. by accessing the result directly afterwards and making sure that the result won't be pre-calculated at compile time. When you benchmark things, make sure that you're getting sensible results and check how they scale with the problem size to make sure that work is actually being done. 

### Loop Unrolling

Loops in C++ are usually directly modelled in the machine code as well, using conditional tests and "jump" statements (essentially `goto` statements). This means that when we have a loop in our code like this:

```cpp
for(int i = 0; i < 8; i++)
{
    v[i] = i;
}
```

it does more than just execute the lines inside the loop, in this case eight assignment statements. At every iteration it has to keep track of the counter `i`, and test the conditional statement `i < 8`, and assign the program counter (this is what tells the CPU which instruction to execute next) to the correct line to jump to depending on the outcome of that test. For loops where the number of iterations can be determined at compile time, this overhead can be eliminated by making copies of the statements inside the loop. This effectively transforms your code into something like this:

```cpp
v[0] = 0;
v[1] = 1;
v[2] = 2;
v[3] = 3;
v[4] = 4;
v[5] = 5;
v[6] = 6;
v[7] = 7;
```
(In practice this change is in the machine code, not the C++!) This can make your code execute faster, but it also makes your final machine code, and therefore the size of your executable on disk, larger.  

### Single Instruction Multiple Data (SIMD)

Modern CPUs typically contain units specially designed for SIMD, or "Single Instruction Multiple Data", workflows. As the name suggests, SIMD refers to performing the same operation on multiple pieces of data at the same time. (This kind of behaviour is done at a much larger scale on accelerated devices like GPUs!)

A typical CPU (x86 architecture or ARM) SIMD register will be 128 bits, meaning that it can operate simultaneously on:

- 2 x `double` (each 64 bits)
- 4 x `float`  (32 bit)
- 4 x `int`    (32 bit)

and might contain 8 or 16 of these registers. Many x86 processors will also have 256-bit or even 512-bit registers; making use of these requires compiling with additional flags since they are less common and therefore the code generated will be less portable.

#### Data Alignment for SIMD

To get the best performance out of SIMD operations we need to consider data _alignment_. When loading, for example, four `float` values into a 16-byte (128-bit) SIMD register, then we load 16 contiguous bytes in memory. The most efficient loading mechanism doesn't just load 16-byte pieces of memory starting at _any_ address, but rather the view of RAM is broken up into 16-byte sections, and you can load any one of these sections quickly. Loading four floats that crosses one of these boundaries is less efficient than being _aligned_ with these boundaries. Luckily we can align our data with specific boundaries in memory using the following syntax:

```cpp
// Aligned stack allocations@
//4 floats aligned to 16 byte boundary
alignas(16) float f16[4];

//24 floats aligned to 32 byte boundary. 
alignas(32) float f32[24];

// Algined heap allocations
// four floats aligned to 16-byte boundary
float *x = new (std::align_val_t(16)) float[4];
// four doubles aligned to 32-byte boundary
double *y = new (std::align_val_t(32)) double[4]; 
```

- For 128-bit registers in x86 and ARM processors you want to be aligned with 16-byte boundaries
- For 256-bit registers as in AVX you want to be aligned with 32-byte boundaries. 
    - 32-byte alignment will also work with 128-bit registers, which means it can allow for efficient vectorisation whether or not you have the larger SIMD registers, but is not necessary if you are worried about packing data as densely as possible in memory. 
- In the example above the first value of each array is aligned to the boundary. If we look at `f32`, which contains 24 floats aligned to a 32-byte boundary, then `f32[0]`, `f32[8]`, and `f32[16]` will all be aligned with 32-byte boundaries, because each float is 4-bytes and therefore there are 8 floats to a 32-byte block. This means that we can efficient load the blocks `f32[0] ... f32[7]`, `f32[8] ... f32[15]`, and `f32[16] ... f32[23]` into registers for SIMD operations. 
- We _can_ load unaligned data into registers for SIMD operations, it just isn't quite as efficient. 

#### SIMD Optimisation and Loop Dependency

Compilers can make use of these kinds of units as long as the calculations are independent - we can't calculate two things in parallel if one depends on the output of the other. Determining whether things are independent in this way, especially when there are loops complicated, is not always trivial so you may not always get the most usage out of these registers. A loop like the following:

```cpp
for(int i = 0; i < 4; i++)
{
    v[i] *= 2;
}
```

should be able to be calculated in a parallelised way, but 

```cpp
for(int i = 0; i < 4; i++)
{
    sum += v[i];
}
```

can't because there is loop dependency. Calculating a sum like this in parallel would require reformulating the problem in a way that the compiler will not do by itself. Remember that **re-ordering floating point operations changes the result** and so many arithmetic processes cannot be automatically vectorised by the compiler, even if the vectorisation appears obvious.

#### Manual SIMD

SIMD can be manually implemented using [C++ intrinsics](https://learn.microsoft.com/en-us/cpp/intrinsics/compiler-intrinsics?view=msvc-170), which map very closely onto specific assembly level instructions. These are easier to use than writing directly in assembly, and can be used to enforce that you get the exact vectorisation strategy that you want, but because of their close relationship with low-level instructions these are not as portable as normal C++ code. x86 and ARM processors have similar functionality for the most part, but a completely different set of intrinsics. In order to write portable code with this kind of approach, programmers usually detect or specify the architecture at build time and use pre-processor directives to determine which intrinsics are used, what data alignments are required, and so on. That however is beyond the scope of this course! 

To see how intrinsics work, consider the example of adding floats in parallel using SIMD. 
1. You load 4 (or 8) floats into a 16 (or 32) byte register from your first memory address.
2. You load 4 (or 8) floats into a 16 (or 32) byte register from your second memory address.
3. You perform a vectorised addition operation.
4. You place the resulting 4 (or 8) floats into your destination memory address.

For x86 we need the commands / includes:
```cpp
    // 128 bit definitions
    #include <xmmintrin.h>

    // 256 bit definitions
    #include <immintrin.h>

    // _mm_load_ps takes a pointer to the first float of a pack of four floats
    __m128 loaded_floats = _mm_load_ps(address);
        
    // vectorised addition, takes 2 __mm128 arguments
    __mm128 result = _mm_add_ps(loaded_x, loaded_y);

    // store value into memory address, takes a pointer and an __mm128
    _mm_store_ps(address, result)

    //256 bit intrinsics
    // load a float buffer. __mm256 i
    __m256 loaded_floats = _mm256_load_ps(address);

    // performing a vector addition
    __mm256 result = _mm256_add_ps(loaded_x, loaded_y)
            
    //store
    _mm256_store_ps(address, result);
```
    
for ARM we need the commands / includes:
    
```cpp
    #include <arm_neon.h>

    // Load data: takes a pointer to the first of the four floats
    // float32x4_t is for 4 floats i.e. 128 bit
    float32x4_t loaded_floats = vld1q_f32(addres);
        
    // Vectorised add: takes 2 float32x4_t type arguments
    float32x4_t result = vaddq_f32(loaded_x, loaded_y);

    // store in memory: takes a pointer and a float32x4_t argument and stores the result at that address
    vst1q_f32(address, result);
```


### Function Inlining 

Function calls also have overheads, since we must look up the function, create a new stack frame for the function's scope, and return from the function. This can be avoided by function inlining, which effectively replaces a function call with the code for that function. (This is a similar process to the loop unrolling we saw earlier.) This can have a number of [pros and cons which are discussed here](https://cplusplus.com/articles/G3wTURfi/), with the main down sides having to do with executable size and bloating function code which can itself cause performance issues. As a result, inlining isn't done manually (we generally can't _force_ a compiler to inline a function; there are keywords to encourage inlining but these are not guaranteed), and the compiler will decide whether a function should be inlined for a given function call. This can be influenced by a number of flags 

- `-fno-inline`: default for non optimised code, does not inline functions unless explicitly marked. 
- `-finline-functions`: consider all functions for inlining. (Does not mean that all functions are inlined!)
- `-finline-small-functions`: consider functions for inlining if inlining will decrease program size i.e. function is smaller than the machine code required to make the function call (function call overheads). 
- And many more in the documentation!

## Optimising Standard Library Objects 

Many of the standard library objects that we use such as smart pointers are more complex than their less safe C-style counterparts (like raw pointers). In unoptimised code this can lead to a substantial performance hit for using these structures, but this does not need to be the case. Under the hood, smart pointers are wrappers for raw pointers that enforce desirable properties. These wrappers often lead to additional function calls for basic things like data accesses, even though the `*` operator works in the same way as their C style counterpart. An optimising compiler can eliminate the middle man in these cases and access data more directly, reducing the overheads of these structures without affecting their safety properties (memory management functionality will still be inserted for example when objects need to be destroyed). 

In general, you shouldn't worry too much about how optimised objects like these are unless you can demonstrate that interacting with them is a bottleneck in your program (e.g. using profiling) and have a strong argument that using a lower level, faster structure is worth the trade-off. 

**Don't worry about using modern C++ features like smart pointers and containers, but do turn on the optimisations for production code!**

## Floating Point Arithmetic 

Floating point arithmetic is how we typically deal with approximating the real numbers in code. Floating point numbers can in principle have any level of precision, but the most common are:

- `float`: 32 bits
- `double`: 64 bits
- `long double`: Usually 80 or 128 bits depending on processor
- `half`: 16 bits, not part of the C/C++ standard!

The data for floating point numbers are split into two parts, the mantissa and the exponent. It is comparable to scientific notation except that it usually uses powers of 2 instead of 10. 

$n = m \times 2^e$

where $m$ is the mantissa and $e$ is the exponent. The mantissa is usually represented so there is a radix point after the first significant (non zero) digit. In other words, the binary `001011` would represent the mantissa `1.011` (again, in base 2). The exponent is a signed integer. 

The size of the data affects both the precision of the mantissa and the range of values for the exponent (and therefore how large and small the values represented can be). 

- `float` has around 7 significant figures
- `double` has around 16 significant figures

Floating point computation **is not exact**. 

- Adding values of very different sizes leads to significant loss of precision since values must be converted to have the same exponent to be added together. This means the difference in scale is pushed into the mantissa, which then loses precision due to leading `0` digits on the smaller number. In some cases the smaller number may be so small that the closest representable number with that exponent is `0` and so the addition is lost completely. 
- Subtracting values which are close in size leads to cancellation of many digits and a result with far fewer significant digits and therefore lower precision. 
- Identities from real arithmetic do not necessarily hold, in particular addition and multiplication are not associative, so $(a + b) + c \neq a + (b + c)$ in floating point! 
- Handling these difficulties in numerical methods is a major field in and of itself. Many numerical algorithms are specially crafted to correct rounding errors in floating point arithmetic.  

### Floating Point Precision

Higher precision floating point numbers (like `double` as opposed to `float`) will give more accurate results when doing numerical work, but may also be slower to perform operations. Historically `double` operations have taken more time to compute than `float` operations, although this is no longer typically the case on modern CPUs. Nevertheless, if you are exploiting SIMD registers for maximal performance, fewer `double` values can fit in an available register and therefore fewer operations can be performed in a given amount of time. Some fast algorithms use single precision `float` or even half precision floating point numbers in areas where the results will not be significantly impacted by this. This is particularly common in areas like statistics and machine learning where the statistical variance is much larger than the precision of the floating point numbers. You must always bear in mind that the use of lower precision floating point types can lead to numerical instability from cancellation errors or division/multiplication by extreme values causing under or overflow. You should always have tests for robustness and precision to check that any compromises made to precision are acceptable. 

### Optimisation of Floating Point Arithmetic

Since floating point computation is not exact, many statements which are mathematically equivalent using real numbers are not equivalent in floating point. As mentioned above, compiler optimisations should not change the meaning of the code - the outcome of a calculation. Floating point operations therefore place limitations on the kinds of optimisations that can be applied, and if performance is a major consideration you should try to write out your floating point operations in the most efficient way possible to begin with. 

Nevertheless, compilers can optimise numerical code for speed by rearranging arithmetic operations, even floating point operations. (Integer arithmetic can be rearranged by the compiler because it is exact.) While C++ optimisations are generally designed not to change the results of the calculations as written there are some, such as those enabled by `-ffast-math`, that allow for rearrangement of arithmetic according to the rules of _real numbers_. This allows for example the rearranging associations. This can be a powerful tool in some cases, not only allowing your numerical code to be rearranged into a more efficient format but also permitting the compiler to make the necessary reorderings of operations for vectorising floating point algorithms. There are however, significant drawbacks to using fast-math optimisations due to the way that it can change the meaning of your program.

Suppose we have a large number $N$ and two much smaller numbers $a$ and $b$ and we want to calculate $N + a + b$. We know from the way that floating point numbers work that adding small and large numbers leads to rounding errors, so the best order to add these number is $(a+b) + N$, to keep the precision of $(a+b)$ and maximise the size of the smaller number that we add to $N$. This is why re-associations in optimisation can cause a problem. Numerical algorithms with factors which compensate for rounding errors can have their error corrections optimised away by fast-math, because for _real_ numbers the error corrections would be zero and the compiler can identify them as redundant! 

**Do not use fast math optimisations for code containing precise numerical methods unless you have very strictly tested it for sufficient accuracy.**

## Inspecting Optimised Code

The process of optimisation, and the optimised executable at the end of it, can feel rather confusing and obscure given that we are not directly in control of it. Nevertheless, we still need to make sure that our debugging and testing of our programs is thorough, and there are ways to get a better understanding of what the compiler has done to our code if we're willing to look at a lower level!

### Debugging and Testing

As we have mentioned already, debugging optimised code can be substantially more difficult than debugging unoptimised code. Generally, optimised code should work in the same way as unoptimised code, and therefore:

- Compile code without optimisations (or with `-Og`) for debugging and most development purposes other than profiling / benchmarking. 
- Turn on appropriate optimisations when compiling for actual deployment. 
- Run your unit tests on both unoptimised and optimised code.
    - If you have tests which pass when unoptimised but fail when optimised, then the chances are you have a bug in your code causing [**undefined behaviour**](https://en.cppreference.com/w/cpp/language/ub). 

### Optional: Dumping Executables and Inspecting Assembly Code

This course is certainly not about assembly or low level programming, but you can understand something about what optimising compilers are doing if you take a look at the end result of your compilation. This can be instructive when working with small examples to better understand what the compiler is doing to your code. 

We can use the command:

```bash 
objdump -d <myexe> >> <output_file>
```

to convert the contents of the executable into a human readable assembly code. 

In order to understand assembly code, we need a basic understanding of [the registers on our CPU](https://en.wikibooks.org/wiki/X86_Assembly/X86_Architecture) and [some typical assembly instructions](https://flint.cs.yale.edu/cs421/papers/x86-asm/asm.html). Values are also written as [hexadecimals](https://en.wikipedia.org/wiki/Hexadecimal) with the prefix `0x`. 

- Registers on the CPU store data which you are working with. These include:
    - Pointers to the bottom and top of the stack. 
    - Registers for arithmetic operations, I/O etc. 
    - Registers for SIMD. 
- We won't have variables or types, just data in memory and in registers. This is much more closely aligned to how the machine actually works!
- The program counter tells the CPU which instruction to read and execute. 
- Data is moved from memory to registers to be worked on and then back to memory again. 
    - This is important to understand how certain race conditions work in shared memory parallel programming, which we'll go over next week! You might want to think about what could happen if two processes both want to work on the same piece of memory and copy it into their own registers to work on.
    - Assembly code also refers to virtual memory for the same reasons as C++, so the physical address and whether the memory is in RAM or cache are still unknown to us and handled by the OS/hardware. 
- Mostly the code deals with memory addresses and registers:
    - A register is denoted with a `%` e.g. `%rax` is the location of the 64-bit accumulator register.
    - The value held at a location is indicated by `()` e.g. `(%rax)` refers to the value in the `rax` register rather than the register itself. 
    - Constants are denoted with `$` e.g `$0x0` is the constant `0`.

We won't go into detail on inspecting assembly code but point out some key things that can be observed:

- Functions will be labelled, and you can easily find your `main` function.
- Optimised code is usually significantly shorter than unoptimised code, especially when using structures like smart pointers where overheads can be optimised away.
- Loops are formed from a combination of `cmp` (compare) and `jmp/jg/jl...` (jump, jump if greater than, jump if less than etc.) statements which test the conditions of the loop and redirect the program counter accordingly. You can therefore see if loops are unrolled or otherwise optimised out, which may lead to longer sections of assembly code. 
- Function calls are made by `call/callq` statements. Unnecessary function calls can be optimised out by inlining or removing redundant functions. 
- [SIMD instructions](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions#SSE_instructions) are recognisable if your compiler has performed this kind of optimisation. 
- If you assign a constant to a variable like `int x = 5`, you can usually spot this by finding the constant `$0x5` in a `mov` statement (`mov` moves a constant or a value contained in a source memory address/register to a destination memory address/register). 

As a simple example consider the code:

```cpp
    int u[100], v[100], w[100];

    for(int i = 0; i < 100; i++)
    {
        u[i] = i*i;
        v[i] = 2*i;
    }

    //Summation loop
    for(int i = 0; i < 100; i++)
    {
        w[i] = u[i] + v[i];
    }
```

My unoptimised code has the following assembly for the summation loop:

```
    123f:	83 bd 38 fb ff ff 63 	cmpl   $0x63,-0x4c8(%rbp)
    1246:	7f 38                	jg     1280 <main+0xb7>
    1248:	8b 85 38 fb ff ff    	mov    -0x4c8(%rbp),%eax
    124e:	48 98                	cltq   
    1250:	8b 94 85 40 fb ff ff 	mov    -0x4c0(%rbp,%rax,4),%edx
    1257:	8b 85 38 fb ff ff    	mov    -0x4c8(%rbp),%eax
    125d:	48 98                	cltq   
    125f:	8b 84 85 d0 fc ff ff 	mov    -0x330(%rbp,%rax,4),%eax
    1266:	01 c2                	add    %eax,%edx
    1268:	8b 85 38 fb ff ff    	mov    -0x4c8(%rbp),%eax
    126e:	48 98                	cltq   
    1270:	89 94 85 60 fe ff ff 	mov    %edx,-0x1a0(%rbp,%rax,4)
    1277:	83 85 38 fb ff ff 01 	addl   $0x1,-0x4c8(%rbp)
    127e:	eb bf                	jmp    123f <main+0x76>
```
- `cmpl   $0x63,-0x4c8(%rbp)` compares the value that holds the iteration variable `i` with `99`
- `jg     1280 <main+0xb7>` jumps past the loop if `i` was greater than `99`
- `jmp    123f <main+0x76>` jumps back to the start of the loop. 
- We add see that we are using the normal integer `add` inside the loop after moving the values from memory to the registers `%eax` and `%edx`. 
- It also adds one (`$0x1`) to the iteration variable each time (`addl`). 
- Basically this works exactly as you would expect it to!

If I optimise my code however, my loop can look quite different:
```
    11e0:	66 0f 6f 04 02       	movdqa (%rdx,%rax,1),%xmm0
    11e5:	66 0f fe 04 01       	paddd  (%rcx,%rax,1),%xmm0
    11ea:	0f 29 04 03          	movaps %xmm0,(%rbx,%rax,1)
    11ee:	48 83 c0 10          	add    $0x10,%rax
    11f2:	48 3d 90 01 00 00    	cmp    $0x190,%rax
    11f8:	75 e6                	jne    11e0 <main+0xa0>
```
- The loop condition is evaluated at the end of the first iteration, and now compares the value in the `%rax` (64-bit accumulator) register with the value `0x190` (400). 
- `jne    11e0 <main+0xa0>` jumps back to the start of the loop if the values in these are not equal. 
- The value in `%rax` which represents the loop iteration variable is now advanced by `0x10` (16), resulting in 25 iterations. This is what we might expect given that we are using an SIMD register which can handle 4 32-bit integers at a time. 
    - Note that although I have multiple SIMD registers, I can still only perform one instruction at a time, which means I can't do more than 4 integer additions at once. 
- We're now making use of SIMD instructions like `movdqa`, `paddd`, and `movaps` for _packed integers_ i.e. multiple integers stored as one string of data.
    - Note that compiled programs will often use the vectorised registers `xmm_` and `ymm_` for sequential floating point arithmetic as well as vectorised arithmetic, so you need to look out specifically for the vectorised arithmetic instructions rather than just what registers are being used. 

There are many other changes made to this program due to the high level of optimisation asked for (`-03`), but this should illustrate the impact that compiler optimisation can have on the actual machine operation, and how we can inspect and understand this. 
