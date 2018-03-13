---
title: Boost
---

## Boost

### Linux Users

Install the appropriate package with apt-get or yum, for example:

``` Bash
sudo apt-get install boost
```

On Ubuntu 13.04 boost is spilt into multiple packages. The package needed here is "libboost1.53-dev" and
to install it you should do:

``` Bash
sudo apt-get install libboost1.53-dev
```


Check that your package manager is delivering at least version 1.53, if you have an earlier version, you will need to download from source, following the windows instructions below.

### Mac Users

``` Bash
brew install boost
```

### Windows Users

You will need to download and install boost manually.

Create an appropriate folder, somewhere near your working code for the course.

For example, if you are working in `/home/myusername/devel/cppcourse/rsd-cppcourse/reactor` you could do:

``` Bash
mkdir -p ~/devel/libraries/boost
cd ~/devel/libraries/boost
```

Now, download and unzip boost, which will take quite a while:

``` Bash
curl -L http://sourceforge.net/projects/boost/files/...
		...boost/1.54.0/boost_1_54_0.zip/download > boost.zip
unzip boost.zip
```

(All one line, not literally two lines with dots!)

### CMake and Boost

If we download from source, before we next build, we will need to tell our shell where the boost library can be found

```Bash
cd build
export CMAKE_INCLUDE_PATH=/home/myusername/devel/libraries/boost_1_54_0
cmake ..
make
ctest
```

... or your equivalent.
