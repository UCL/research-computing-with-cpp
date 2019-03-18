---
title: Python Binding
---

## Live Example


### Overview

* [CMakeCatchTemplate][CMakeCatchTemplate] also has Python examples
* Use either [BoostPython][BoostPython] or [PyBind11][PyBind11]
* PyBind11 is header only, should be easier
* But Boost seems to have better examples on the internet


### Demo / Code Walk Through

We shall step through a simple function:

* Adds 2 integers
* [Boost.Python example][BoostPythonExample]
* [PyBind11 example][PyBind11Example]


### Other examples

* OpenCV uses numpy ndarray (Python) and converts to ```cv::Matt``` (C++)
* See [Gregory Kramida's pyboostconverter][pyboostconverter], integrated [here][CMakeCatchTemplateOpenCV], using Boost.Python.
* Also see ongoing C++/OpenCV project: [scikit-surgeryopencvcpp][scikit-surgeryopencvcpp], using Boost.Python, and [Gregory Kramida's pyboostconverter][pyboostconverter]
* Also see ongoing C++/[PCL][PCL] project: [scikit-surgerypcl][scikit-surgerypcl], using Boost.Python, Boost.Numpy and [PCL][PCL].


[CMakeTemplate]: https://github.com/MattClarkson/CMakeCatchTemplate
[BoostPython]: https://www.boost.org/doc/libs/1_68_0/libs/python/doc/html/index.html
[PyBind11]: https://github.com/pybind/pybind11
[BoostPythonExample]: https://github.com/MattClarkson/CMakeCatchTemplate/blob/master/Code/PythonBoost/mpLibPython.cpp
[PyBind11Example]: https://github.com/MattClarkson/CMakeCatchTemplate/blob/master/Code/PythonPyBind/mpLibPython.cpp
[pyboostconverter]: https://github.com/Algomorph/pyboostcvconverter
[CMakeCatchTemplateOpenCV]: https://github.com/MattClarkson/CMakeCatchTemplate/blob/master/Code/PythonBoost/mpLibPythonWithOpenCV.cpp#L34
[scikit-surgeryopencvcpp]: https://github.com/UCL/scikit-surgeryopencvcpp
[scikit-surgerypcl]: https://github.com/UCL/scikit-surgerypclcpp
[PCL]: http://pointclouds.org/