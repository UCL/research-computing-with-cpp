---
title: Choosing Libraries
---

## Choosing libraries

### Libraries are awesome

[MPHYG001][Python04Intro]: The strength of a language lies as much in the set of libraries available, as it does
in the language itself.

A great set of libraries allows for a very powerful programming style:

* Write minimal code yourself
* Choose the right libraries
* Plug them together
* Create impressive results

### Libraries for Efficiency

Not only is this efficient with your programming time,
it's also more efficient with computer time.

The chances are any general algorithm you might want to use
has already been programmed better by someone else.


### Licensing - Just Including A 3rd Party Library

CAVEAT: This is not legal advice. If in doubt, seek your own legal advice.

* If you use your code for internal use and you don't distribute it
    * then you are ok.
    * i.e. if 3rd party library wasn't for use, then it wouldn't be available.
    * you can publish results using open-source code
    * however, increasingly you are asked to share code ... read on.


### Licensing - Choosing Yours

CAVEAT: This is not legal advice. If in doubt, seek your own legal advice.

* However, you may plan to distribute your code:    
    * Read [this book][LicensingBook], and/or [GitHub's advice][Chooselicense], and [OSI][OSI] for choosing your own license.
    * Don't write your own, unless you use legal advice.
    * Try to pick one of the standard ones if you can, so your software is "compatible", and people understand the restrictions.


### Licensing - Redistribution

* If you distribute your code, the licenses of any 3rd party libraries take effect:
    * [MIT][MITLicense] and [BSD][BSDLicense] are permissive. So you can do what you want, including sell it on.
    * [Apache][ApacheLicense] handles multiple contributors and patent rights, but is basically permissive.
    * [GPL][GPLLicense] requires you to open-source your code, including changes to the library you imported, and your work is considered a "derivative work", so must be shared.
    * [LGPL][LGPLLicense] for libraries, but use dynamic not static linkage. If you use static linking its basically as per [GPL][GPLLicense].

Note: Once a 3rd party has your code under a license agreement, their restrictions are determined by
that version of the code.


### Licensing - So Think Long Term

* On [MPHYG001][PythonCourse], we encouraged
    * Share your code, collaborate, take pride.
    * This improves your code and your science. (See [this][NatureArticle]).
    * Your software should accumulate, reliably, and be extensible.
* So, take care in your choice of 3rd party library
    * Don't want to redo work later, at the end of PhD.
    * Don't want to rely too heavily on non-distributable code.
    * But if you do, understand what that means.


### Developer Community - 1

* Code that is not developed, rots.
    * When was the last commit?
    * How often are there commits?
    * Is the code on an open version control tool like GitHub?

### Developer Community - 2

* Can you find the lead contributor on the internet?
    * Do they respond when approached?
* Are there contributors other than the lead contributor?


### Developer Community - 3

* Are there many unit tests, do they run, do they pass?
* Does the library depend on other libraries?
* Are the build tools common?
* Is there a sensible versioning scheme (e.g. [semantic versioning][semver]).
* Is there a suitable release schedule?


### Implementation Details

* Shouldn't need to look excessively closely, but consider
    * Documentation
    * Number of ToDo's
    * Dependencies
    * Data Structures? How to import/export image
    * Can you write a convenient wrapper?
    * Then look at features


### Summary

* In comparison with languages such as Python
    * In C++ prefer few well chosen libraries
    * Be aware of their licenses for future distribution
    * Keep a log of any changes, patches etc. that you make
    * Be able to compile all your own code, including libraries
        * so need common build environment. (eg. CMake, Make, Visual Studio).

[PythonCourse]: http://development.rc.ucl.ac.uk/training/engineering
[Python04Intro]: http://development.rc.ucl.ac.uk/training/engineering/session04/
[NatureArticle]: http://www.nature.com/news/2010/101013/full/467753a.html
[LicensingBook]: http://www.oreilly.com/openbook/osfreesoft/book/
[Chooselicense]: http://choosealicense.com/
[OSI]: http://opensource.org/
[MITLicense]: http://opensource.org/licenses/MIT
[BSDLicense]: http://opensource.org/licenses/BSD-3-Clause
[ApacheLicense]: http://opensource.org/licenses/Apache-2.0
[GPLLicense]: http://opensource.org/licenses/gpl-license
[LGPLLicense]: http://opensource.org/licenses/lgpl-license
[semver]: http://www.semver.org/
[NifTK]: http://www.niftk.org/
