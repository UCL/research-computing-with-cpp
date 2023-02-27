---
title: Why Optimise for Performance?
---

Estimated Reading Time: 10 minutes

# Performance Optimisation Considerations

## What Should You Optimise and Why? 

Performance optimisation has always been a significant concern in scientific computing, although the motivations, and targets of optimisation, can vary. In principle we usually think about optimisation of two different kinds:

- Time: we want to optimise algorithms for speed to get the same calculation done more quickly. 
- Space: we want to optimise algorithms for memory usage to get the same calculation done with lower memory requirements. 

In modern physics the need for efficient data processing is pressing: 

- Large simulations for cosmology, fluid dynamics, or materials science are extremely computationally intensive, running for days, weeks, or months at a time. 
- [LSST](https://www.lsst.org/about/dm), a cosmological observatory, will receive about 20TB of data _per day_. 
- Engineering applications often have to be responsive in real time. 
- Many systems are highly restricted - for example satellites which have to be launched into space - and therefore need to get the most work possible out of their hardware. 

Regardless of your particular field, high performance computing is often desirable, allowing us to explore more detailed and advanced models than we could normally. As researchers, it's our job to push those boundaries! 

## When To Start Optimising Code

When faced with writing code that one knows needs to be efficient, it is common to try to optimise every piece of code they write as one goes along. This can lead to a very slow development process, as well as difficult to understand software! There are some key points to bear in mind when you are developing efficient code:

- Optimisation often takes significant work: it is much slower to write a highly optimised implementation than a straight-forward one.
- Optimised code is often unintuitive and therefore harder to understand for new users. This also makes maintenance harder, especially if it is not obvious what can and can't be changed without breaking the optimisation benefits! 
- Highly optimised code may not be portable, as code may be optimised for specific platforms, architectures, or even individual machines.
- KISS (Keep It Simple, Stupid) is a good principle for first drafts: focus on writing _correct_, understandable code first. 
- Measure performance to find your bottlenecks: don't waste time optimising parts of your program that only make up a tiny fraction of your runtime (or memory). Chances are there are only one or two places where a substantial amount of your runtime is spent and which can be effectively optimised. **Profile your program and prioritise high-impact optimisations first.** 
- Continue measuring performance to check that you are making an impact on your runtime / memory usage.
- Monitor your optimised algorithms for **accuracy and precision**. Some algorithms may produce more reliable results than others, especially when using floating point arithmetic. A significant consideration with numerical algorithms is not just the time and space usage, but the numerical stability of the algorithm. 
    - For example some methods use single (or even half) precision floating point numbers instead of doubles in order to get faster results. This sometimes happens in some statistical methods such as MCMC (Markov Chain Monte Carlo) where we want to be able to sample a random distribution and the additional precision doesn't have a strong impact on the overall statistical properties.
    - Some methods use arbitrary precision integer arithmetic to handle very large values and do calculations without e.g. cancellation errors, even though this requires more time and memory. This is common in things like financial applications where all the sums really need to add up exactly! 

**Optimisation is usually a trade off between different factors. You should always bear in mind your priorities when creating efficient software.** 

## What Forms Can Optimisation Take?

Optimisation can take a variety of forms: 

- Time / Space efficient algorithms (reduced computational complexity)
- Exploiting specialised hardware (e.g. vectorised arithmetic units, caches)
- Automated optimisation by compilers
- Concurrency (see the next three weeks!)