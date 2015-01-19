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
    
    
### Turing Complete

* Given: A [Turing Machine][TuringMachine]
    * Tape, head, states, program, etc.
* A language is "Turing Complete" if it can simulate a Turing Machine
    * e.g. Conditional branching, infinite looping
* Turing's work underpins much of "what can be computed" on a modern computer
    * C, C++ no templates, C++ with templates, C++ TMP
    * All Turing Complete
* Interesting that compiler can generate such theoretically powerful code.    
* But when, where, why, how to use TMP?    
    
    
### Why Use It?

* Use sparingly as code difficult to follow
* Use for
    * Optimisations 
    * Represent Behaviour as a Type
    * Traits classes
    * Examples provided here
* But when you see it, you need to understand it!


### Factorial Example

See [Wikipedia Factorial Example][TMPWikipedia]

* This:
{{cppfrag('02','TMPFactorial/TMPFactorial.cc')}}

* Produces:
{{execute('02','TMPFactorial/TMPFactorial')}}


### Factorial Notes:

* Compiler must know values at compile time
    * i.e. constant literal or constant expression
    * See also [constexpr][C++11constexpr]
* Generates/Instantiates all functions recursively
* Factorial 16 = 2004189184
* Factorial 17 overflows
* This simple example to illustrate "computation"
* But when is TMP actually useful?
* Notice that parameter was an integer value ... not just "int"
    

### Loop Example

* This:
{{cppfrag('02','TMPLoopUnrolling/TMPLoop.cc')}}

* Time: numberOfInts=3 took 37 seconds


### Loop Unrolled

* This:
{{cppfrag('02','TMPLoopUnrolling/TMPLoopUnrolled.cc')}}

* Time: numberOfInts=3 took 10 seconds  


### Policy Checking

* Stuff


### Traits

* Stuff


### Assertions

* Stuff


### Use in Medical Imaging

* Stuff

[TMPWikipedia]: http://en.wikipedia.org/wiki/Template_metaprogramming
[TMPWikibooks]: http://en.wikibooks.org/wiki/C%2B%2B_Programming/Templates/Template_Meta-Programming
[TMPKeithSchwarz]: http://www.keithschwarz.com/talks/slides/tmp-cs242.pdf
[TuringMachine]: http://en.wikipedia.org/wiki/Turing_machine
[TuringComplete]: http://en.wikipedia.org/wiki/Turing_completeness
[C++11constexpr]: http://en.wikipedia.org/wiki/C%2B%2B11#constexpr_.E2.80.93_Generalized_constant_expressions

