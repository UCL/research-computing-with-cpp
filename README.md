research-computing-with-cpp
===========================

Deployed at:

http://rits.github-pages.ucl.ac.uk/research-computing-with-cpp

To build and test:

``` bash
git clone https://github.com/UCL-RITS/research-computing-with-cpp
cd research-computing-with-cpp
./build.sh
```

You will need to have a bunch of stuff installed in order for the build to succeed.
For Mac:
* Libraries:
   * `brew install boost`
* Ruby stuff:
   * `brew install ruby`
   * `gem install liquid`
   * `gem install jekyll`
   * `gem install redcarpet`

Then in folder _site, you'll have the `html`'s.
Or, for a shortcut: `make preview`

See https://github.com/UCL-RITS/research-computing-with-cpp/blob/master/01cpp/index.md for an example of how to reference a C file, CMake file, and run an executable

And https://github.com/UCL-RITS/research-computing-with-cpp/tree/master/01cpp/cpp/hello
for the corresponding code
