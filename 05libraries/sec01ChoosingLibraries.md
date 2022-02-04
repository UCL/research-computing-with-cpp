---
title: Choosing Libraries
---

## Choosing libraries

### Libraries are Awesome

A [great set of libraries allows for a very powerful programming style][Python04Intro]:

* Write minimal code yourself
    * Choose the right libraries
    * Plug them together
    * Create impressive results


### Libraries for Efficiency

Not only is this efficient with your programming time,
it's also more efficient with computer time.

The chances are any general algorithm you might want to use
has already been programmed better by someone else.


### Software Licenses

**CAVEAT**: This is not legal advice. If in doubt, seek your own legal advice (e.g., [UCL Copyright advice][UCL Copy]).

A software license is a way to grant permissions for use and/or distribution to others.
Putting your code on a public website does not grant permissions to anyone to use, copy, translate or distribute it as
you still owns the copyright of that work (even when it's not stated that you own the copyright).

> Contrary to popular belief, distributed unlicensed software (not in the public domain) is fully copyright protected, and therefore legally unusable (as no usage rights at all are granted by a license) until it passes into public domain after the [copyright term][cterm] has expired.[3]
>
> -- [Software Licenses][SL-wiki] -- Wikipedia 2021

You need to consider both:

* License for 3rd party code / dependencies you are using, and
* License for your code when you distribute it.

Remember: even if you aren't distributing code yet, you need to understand the licenses of your dependencies.


#### Third Party Licenses

When you distribute your code, the licenses of any libraries you use takes effect.
For example, a library with license:

* [MIT][MITLicense] or [BSD][BSDLicense] are permissive. So you can do what you want, including sell it on.
* [Apache][ApacheLicense] handles multiple contributors and patent rights, but is basically permissive.


#### Third Party Licenses

* When you distribute your code, the licenses of any libraries you use takes effect
* If library has:
    * [GPL][GPLLicense] requires you to open-source your code, including changes to the library you imported, and your work is considered a "derivative work", so must be shared.
    * [LGPL][LGPLLicense] for libraries, but [use dynamic not static linkage][LGPLStaticVsDynamic]. If you use static linking its basically as per [GPL][GPLLicense].

However, there's still some debate on [GPL/LGPL and derivative works](https://lwn.net/Articles/548216/) - only true test is in court.


#### Choosing a License

* When you plan to distribute code:
    * Don't write your own license, unless you use legal advice and you understand its consequences.
    * Check [GitHub's advice][Chooselicense], and [OSI][OSI] for choosing your own license.
    * Try to pick one of the standard ones for [compatibility][LicesnseCompatibilityWiki].

For an in-depth understanding we recommend you read some works about licenses:

- [Understanding Open Source and Free Software Licensing][LicensingBookWC] ([web][LicensingBook],[UCL][LicensingBookUCL],[pdf][LicensingBookPDF]),
- [The public domain][PDBook]

**Note**: Once a 3rd party has your code under a license agreement, their restrictions are determined by that version of the code.

     
### Choose Stability

* So, take care in your choice of 3rd party library
    * Don't want to redo work later.
    * Don't want to rely too heavily on non-distributable code.
    * But if you do, understand what that means.


### Is Library Updated?

Code that is not maintained rots!

* When was the last commit?
* How often are there commits?
* Is the code developed on the open (e.g., on GitHub)?
* Is there a sensible versioning scheme (e.g. [semantic versioning][semver])?
* Is there a suitable release schedule?
* A [changelog][changelog] is provided which each new release?


### Are the developers reachable?

* Does the "project" offer one or more communication channels?
    * mailing lists, community forums, chat rooms
    * Do they respond when approached?
* If not, can you find the lead contributor on the internet?
* Are there contributors other than the lead contributor?


### Is the code tested?

* Are there many unit tests, do they run, do they pass?
    * Are they run automatically (i.e., Continuous Integration)?
* Does the library depend on other libraries?
* Are the build tools common?


### Is the library of High Quality?

Shouldn't need to look excessively closely, but consider:

* Documentation: does it exists? is it good?
* Number of ToDos: do they keep a track of bugs to fix and future features to implement?
* Dependencies: does it offer a clear list of dependencies? Are they trusted? (i.e., recursively)
* Data Structures: is it clear how to import/export data or images to use later?
* Clear API: can you write a convenient wrapper?
    
    
### Library Features

Then look at features like:

* Manual, Tutorials, ...
* Easy to use


## Summary

* In C++ prefer few well chosen libraries (the work to replace them may be harder than in other languages such as MATLAB/Python)
* Be aware of their licenses for future use and distribution
* Keep a log of any changes, patches, etc. that you make
* Be able to compile all your own code, including libraries
    * so need common build environment. (eg. CMake, Make, Visual Studio).

[PythonCourse]: http://github-pages.ucl.ac.uk/rsd-engineeringcourse/
[Python04Intro]: http://github-pages.ucl.ac.uk/rsd-engineeringcourse/ch04packaging/01Libraries.html
[NatureArticle]: http://www.nature.com/news/2010/101013/full/467753a.html
[LicensingBook]: http://www.oreilly.com/openbook/osfreesoft/book/
[Chooselicense]: http://choosealicense.com/
[OSI]: http://opensource.org/licenses
[MITLicense]: http://opensource.org/licenses/MIT
[BSDLicense]: http://opensource.org/licenses/BSD-3-Clause
[ApacheLicense]: http://opensource.org/licenses/Apache-2.0
[GPLLicense]: http://opensource.org/licenses/gpl-license
[LGPLLicense]: http://opensource.org/licenses/lgpl-license
[semver]: http://www.semver.org/
[cterm]: https://en.wikipedia.org/wiki/Copyright_term
[3]: https://blog.codinghorror.com/pick-a-license-any-license/
[SL-wiki]: https://en.wikipedia.org/w/index.php?oldid=999552724#Software_licenses_and_copyright_law
[UCL Copy]: https://www.ucl.ac.uk/library/ucl-copyright-advice "UCL Copyright advice"
[LGPLStaticVsDynamic]: https://www.gnu.org/licenses/gpl-faq.en.html#LGPLStaticVsDynamic "Does the LGPL have different requirements for statically vs dynamically linked modules with a covered work?"
[LicensingBookWC]: https://www.worldcat.org/title/understanding-open-source-free-software-licensing-guide-to-navigating-licensing-issues-in-existing-new-software/oclc/314704943
[LicensingBookUCL]: "https://ucl-new-primo.hosted.exlibrisgroup.com/primo-explore/search?query=any,contains,9780596005818&tab=local&search_scope=CSCOP_UCL&vid=UCL_VU2"
[LicensingBookPDF]: https://people.debian.org/~dktrkranz/legal/Understanding%20Open%20Source%20and%20Free%20Software%20Licensing.pdf
[LicesnseCompatibilityWiki]: https://en.wikipedia.org/wiki/License_compatibility
[PDBook]: https://www.thepublicdomain.org/download/
[changelog]: https://keepachangelog.com/en/1.0.0/
