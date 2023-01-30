---
title: CMakeCatchTemplate
---

## CMakeCatchTemplate

### Intro 

* Demo project on [GitHub](https://github.com): [CMakeCatchTemplate](https://github.com/MattClarkson/CMakeCatchTemplate)
* No functional code, other than adding 2 numbers
* Basically shows how to use CMake, via various examples.


### Features

* Full feature list in [README.md](https://github.com/MattClarkson/CMakeCatchTemplate/blob/master/README.md)
* SuperBuild:
    * Downloads Boost, Eigen, glog, gflags, OpenCV, PCL, FLANN, VTK
* Example GUI apps (beyond scope of course)
* Unit testing


### SuperBuild

* See flag: ```BUILD_SUPERBUILD:BOOL=[ON|OFF]```
* If ```OFF```
    * Just compiles *this* project in current folder
* If ```ON```
    * Dependencies in current folder
    * Compiles *this* project in sub-folder
* Try it, in separate build folders.


### Homework - 10 

Setup folders like:

```bash
# Source folder
/User/me/build/CMakeCatchTemplate

# SuperBuild folder
/User/me/build/CMakeCatchTemplate-SuperBuild
```

* Ensure ```BUILD_SUPERBUILD=ON```
* Try turning a small library like ```BUILD_gflags``` to ON
* Run the build in the ```SuperBuild``` folder
* Look on disk, see how the dependencies are compiled in the ```CMakeCatchTemplate-SuperBuild``` folder.
* Then look in sub-folder ```MYPROJECT-build``` to see how this project is build, using the dependencies in the folder above.
