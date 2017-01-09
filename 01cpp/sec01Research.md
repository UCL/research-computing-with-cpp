---
title: C++ in Research
---

## C++ In Research

### Problems In Research

* Software is generally poor quality
* I'm not a software engineer
* I don't have time
* I'm unsure of my code


### Solutions for Research

* I'm not a software engineer
    * Learn effective, minimal tools
* I don't have time
    * Unit testing, choose your battles
* I'm unsure of my code
    * Share, collaborate


### C++ Disadvantages

Some people say:

* Compiled language (compiler versions, libraries, platform specific etc)
* Perceived as difficult, error prone, wordy, unfriendly syntax
* Result: It's more trouble than its worth?


### C++ Advantages

* Fast, code compiled to machine code
* Nice integration with CUDA, OpenACC, OpenCL, OpenMP, OpenMPI
* Stable, evolving standard, powerful notation, improving
* Lots of libraries, Boost, Qt, VTK, ITK etc.
* Result: Good reasons to use it, or you may *have* to use it


### Research Programming

* Software is Expensive
    * Famous Book: [Mythical Man Month](http://www.amazon.co.uk/Mythical-Man-month-Essays-Software-Engineering/dp/0201835959/ref=sr_1_1?ie=UTF8&qid=1452507457&sr=8-1&keywords=mythical+man+month)
    * Famous People: [Edsger W. Dijkstra](https://www.cs.utexas.edu/users/EWD/)
* In research we don't know what the end product is
* We need to test lots of ideas quickly


### What about Methodology?

* [Waterfall](https://en.wikipedia.org/wiki/Waterfall_model)
* [Agile](https://en.wikipedia.org/wiki/Agile_software_development)
* At the 'concept discovery' stage, probably too early to talk about product development


### Types of Code

* Divide code:
    * Your algorithm
    * Testing code
    * Data analysis
    * User Interface
    * Glue code
    * Deployment code
    * Paper production
    
    
### Ask Advice

* Before contemplating a new piece of software
    * Ask advice - [Slack Channel](https://ucl-programming-hub.slack.com/)
    * Review libraries to use, and how easy to adopt?
    * Read [Libraries](http://development.rc.ucl.ac.uk/training/engineering/ch04packaging/01Libraries.html) section from [Software Engineering](http://development.rc.ucl.ac.uk/training/engineering/) course
    * Ask about best practices


### Maximise Your Value

* Developer time is expensive
* Your brain is your asset
* Write as little code as possible
* Focus tightly on your hypothesis

(i.e. write the minimum that produces a paper, and do it in the right language)

Don't fall into the trap "Hey, I'll write a framework for that"


### Fame and Glory?

As a researcher I need:

* Reputation in a given field
    * Open-source software, helps citations
    * Smallest possible software package
* Focus on the main algorithmic content of your papers
* Usages/deployment scenarios/packaging/data-analysis all vary    

Examples: [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg), [matching paper](http://www.sciencedirect.com/science/article/pii/S0169260709002533), 300 citations in 5 years.


### What Isn't This Course?

We are NOT suggesting that:

* C++ is the solution to all problems.
* You should write all parts of your code in C++.
* Again - this course is not Beginner C++


### What Is This Course?

We aim to:

* Improve your C++ (and associated technologies).
* Do High Performance Computing (HPC).
* Apply it to research in a pragmatic fashion.
* You need the right tool for the job.
