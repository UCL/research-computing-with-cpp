research-computing-with-cpp
===========================

Deployed to staging at:

http://staging.development.rc.ucl.ac.uk/training/rcwithcpp

Deployed to production at:

http://development.rc.ucl.ac.uk/training/rcwithcpp

To build and test:

cd <development area>
git clone https://github.com/jamespjh/indigo
cd indigo
git checkout dexy
cd ..
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
