---
title: C++ in Research
---

## C++ In Research

### Problems In Research

* Poor quality software
* Excuses
    * I'm not a software engineer
    * I don't have time
    * I'm unsure of my code


### C++ Disadvantages

Some people say:

* Compiled language 
    * (compiler versions, libraries, platform specific etc)
* Perceived as difficult, error prone, wordy, unfriendly syntax
* Result: It's more trouble than its worth?


### C++ Advantages

* Fast, code compiled to machine code
* Nice integration with CUDA, OpenACC, OpenCL, OpenMP, OpenMPI
* Stable, evolving standard, powerful notation, improving
* Lots of libraries, Boost, Qt, VTK, ITK etc.
* Result: Good reasons to use it, or you may *have* to use it


### Research Programming

* Software is already expensive
    * Famous Book: [Mythical Man Month](http://www.amazon.co.uk/Mythical-Man-month-Essays-Software-Engineering/dp/0201835959/ref=sr_1_1?ie=UTF8&qid=1452507457&sr=8-1&keywords=mythical+man+month)
    * Famous People: [Edsger W. Dijkstra](https://www.cs.utexas.edu/users/EWD/)
* Research programming is different
    * What is the end product?


### Development Methodology?

* Will software engineering methods help?
    * [Waterfall](https://en.wikipedia.org/wiki/Waterfall_model)
    * [Agile](https://en.wikipedia.org/wiki/Agile_software_development)
* At the 'concept discovery' stage, probably too early to talk about product development


### Approach

* What am I trying to achieve?
* How do I maximise my output?
* What is the best pathway to success?
* How do I de-risk my research?


### 1. Types of Code

* What are you trying to achieve?
* Divide code:
    * Your algorithm: [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg)
    * Testing code
    * Data analysis
    * User Interface
    * Glue code
    * Deployment code
    * Scientific [paper](http://www.sciencedirect.com/science/article/pii/S0169260709002533) production
     
Examples: NiftyReg 300 citations in 5 years!
    
    
### 2. Maximise Your Value

* Developer time is expensive
* Your brain is your asset
* Write as little code as possible
* Focus tightly on your hypothesis
* Write the minimum code that produces a paper

Don't fall into the trap "Hey, I'll write a framework for that"


### 3. Ask Advice

* Before contemplating a new piece of software
    * Ask advice - [Slack Channel](https://ucl-programming-hub.slack.com/)
    * Review libraries and use them.
    * Check libraries are suitable, and sustainable.
    * Read [Libraries](http://development.rc.ucl.ac.uk/training/engineering/ch04packaging/01Libraries.html) section from [Software Engineering](http://github-pages.ucl.ac.uk/rsd-engineeringcourse/) course
    * Ask about best practices


### Example - NiftyCal

* We should: Practice What We Preach
* Small, algorithms
* Unit tested
* Version controlled
* Small number of libraries
* Increased research output


### Debunking The Excuses

* I'm not a software engineer
    * Learn effective, minimal tools
* I don't have time
    * Unit testing to save time
    * Choose your battles/languages wisely
* I'm unsure of my code
    * Share, collaborate


### What Isn't This Course?

We are NOT suggesting that:

* C++ is the solution to all problems.
* You should write all parts of your code in C++.


### What Is This Course?

We aim to:

* Improve your C++ (and associated technologies).
* Do High Performance Computing (HPC).

So that:

* Apply it to research in a pragmatic fashion.
* You use the right tool for the job.
