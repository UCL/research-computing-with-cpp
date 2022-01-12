research-computing-with-cpp
===========================

Deployed at:

http://rits.github-pages.ucl.ac.uk/research-computing-with-cpp

We normally build these locally on a Mac. We use g++ installed via homebrew. 
So, for g++ version 7, we would: 

``` bash
git clone https://github.com/UCL-RITS/research-computing-with-cpp
cd research-computing-with-cpp
CC=gcc-7 CXX=g++-7 ./build.sh
```

The explicit compiler selection is needed on Mac OS for OpenMP (and possibly MPI)
examples to build. Update the version number as necessary.

You will need to have a bunch of stuff installed in order for the build to succeed.
For Mac:
* Libraries:
   * `brew install boost`
* For building MPI and OpenMP examples:
   * `brew install open-mpi`
   * `brew install gcc`
* Ruby stuff:
   * `brew install ruby`
   * `gem install liquid`
   * `gem install jekyll`
   * `gem install redcarpet`
* Other utilities:
   * `brew install wget`
* Python libraries:
   * `matplotlib` (plus several other scientific python libraries)

Then in folder _site, you'll have the `html`'s.
Or, for a shortcut: `make preview`

See https://github.com/UCL-RITS/research-computing-with-cpp/blob/master/01cpp/index.md for an example of how to reference a C file, CMake file, and run an executable

And https://github.com/UCL-RITS/research-computing-with-cpp/tree/master/01cpp/cpp/hello
for the corresponding code
