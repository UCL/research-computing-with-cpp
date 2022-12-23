research-computing-with-cpp
===========================

Deployed at:

http://github-pages.ucl.ac.uk/research-computing-with-cpp


How to build the lessons locally
-----------------------------------

Clone and build the various output files (that step will need some dependencies)
``` bash
git clone https://github.com/UCL-RITS/research-computing-with-cpp
cd research-computing-with-cpp
./build.sh
```

Then you can proceed to build the website locally. The easiest is via docker.

```bash
mkdir ../bundle # To don't download every single time the ruby dependencies
docker run --rm --volume="$PWD:/srv/jekyll" --volume="$PWD/../bundle:/usr/local/bundle" -it jekyll/jekyll:4 jekyll build
python -m http.server -d _site
```

Dependencies
-------------

All the dependencies are set in the `.github/{texlive,python}/requirements.txt` for an ubuntu machine.

On a Mac, we use g++ installed via [homebrew](https://brew.sh/). So, for g++ version 7, we would run the build command as:

``` bash
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
   * `gem install jekyll --version '~> 4'`
   * `gem install kramdown --version '~> 2.3.1'
   * `gem install jekyll-remote-theme`
`
* Other utilities:
   * `brew install wget`
* Python libraries:
   * `matplotlib` (plus several other scientific python libraries)

A full LaTeX distribution needs to be available to generate a PDF version of the course notes by the build script.

Then in folder _site, you'll have the `html`'s.
Or, for a shortcut: `make preview`



See https://github.com/UCL-RITS/research-computing-with-cpp/blob/master/01cpp/index.md for an example of how to reference a C file, CMake file, and run an executable

And https://github.com/UCL-RITS/research-computing-with-cpp/tree/master/01cpp/cpp/hello
for the corresponding code
