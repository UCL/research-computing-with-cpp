---
title: Why Care About Code Quality?
---

Programming is not just about producing code that works or passes tests. If code is not also usable, it will not be used. If code is not also maintainable, it will become unusable. There are many aspects to code quality, the importance of which depends on the intended use of the code. A very dry list of code quality measures can be found in [Steve McConnell's *Code Complete*][code-complete]:
- External qualities
  - Correctness
  - Accuracy
  - Reliability
  - Robustness
  - Efficiency
  - Adaptability
  - Usability
- Internal qualities 
  - Maintainability
  - Flexibility
  - Portability
  - Reusability
  - Readability
  - Testability
  - Understandability

Steve splits the list into external qualities, which are important to the *user* of the code, and internal qualities, which are important to the programmers who contribute to the code. Let's look at each of these in detail.

### External Qualities

**Correctness**

Does the program do what it is meant to?

**Accuracy**

Are the results close enough to what I need?

**Reliability**

Are the results the same every time the program is run?

**Robustness**

Does the program handle unexpected inputs correctly?

**Efficiency**

Is the program fast enough?

**Adaptability**

Can I extend the program to do something similar but unintended?

**Usability**

Is the program easy to setup and use?

### Internal qualities

**Maintainability**

Are bugs easy to find and to fix?

**Flexibility**

Can I easily add new features to the code?

**Portability**

Am I able to run the program on many different architectures and operating systems?

**Reusability**

Can I use parts of the code in many different places?

**Testability**

Is the code written and designed in such a way that it's easy to test?

**Readability**

How much time do I spend trying to read the code on a surface-level?

**Understandability**

How much time do I spend trying to understand the code?

Some of these qualities are related; readability and understandability are very much entwined, as are correctness and accuracy. Being able to think about code from the perspectives of all these qualities gives you, as a programmer, a much better understanding of what it means to write good code. In saying that, codes designed for different purposes may have very different priorities when it comes to quality. For example, the software controlling a nuclear power station must be extremely correct, robust and reliable, but it may not need to be particularly portable, adaptable or even efficient. On the other hand, a game on you phone can prioritise usability and portability at the expense of some maintainability, robustness and even correctness (if it crashes or glitches, does it matter *that* much?).

### Exercise

> There is no such thing like perfect code. Code can only be good enough.

For each of the following pieces of software, pick one internal and one external quality measure you think is the most important.

- World of Warcraft
- A scientific simulation
- Whatsapp messenger
- A banking app
- The Eigen library


[code-complete]: https://learning.oreilly.com/library/view/code-complete-second/0735619670/
