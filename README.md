research-computing-with-cpp
===========================

[![Join the chat at https://gitter.im/UCL-RITS/research-computing-with-cpp](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/UCL-RITS/research-computing-with-cpp?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Deployed to staging at:

http://staging.development.rc.ucl.ac.uk/training/rcwithcpp

Deployed to production at:

http://development.rc.ucl.ac.uk/training/rcwithcpp

To build and test:

``` bash
cd <development area>
git clone https://github.com/UCL-RITS/indigo-dexy.git indigo
mkdir training
cd training
git clone https://github.com/UCL-RITS/research-computing-with-cpp
cd research-computing-with-cpp
cd build
cmake ..
make
cd ..
dexy
dexy serve
```

See https://github.com/UCL-RITS/research-computing-with-cpp/blob/master/session01/index.md for an example of how to reference a C file, CMake file, and run an executable

And https://github.com/UCL-RITS/research-computing-with-cpp/tree/master/session01/cpp/hello
for the corresponding code

Dexy helper macros are defined at https://github.com/UCL-RITS/research-computing-with-cpp/tree/master/macros

You can also do `{{cppfrag('01','hello/hello.cc','constructor')}}` to link to a particular part of a file
Labelled in the code with ` \\\constructor `
(Triple comment symbol tells dexy this starts a section)
