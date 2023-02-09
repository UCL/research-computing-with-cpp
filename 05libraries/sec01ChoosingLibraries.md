---
title: Choosing Libraries
---

## Choosing libraries

Choosing the right library for the job can be tricky. The right choice can supercharge your development, making it easy to write what you want. However, the wrong choice can set you back a significant amount of time and effort if you end up needing to change a library part-way into a project.

Fundamentally there are two questions to ask that will decide if you **can** use a given library in your project, and a few more that will help you decide if you **should** use the library.

**Can I use the library?**

- Does it provide what I need?
- Can I legally use it?

**Should I use the library?**

- Is the library stable?
- Is the library fast enough for my needs?
- Is the library consistently updated?
- Who develops the library?
- Is the library well-tested?
- Is the library high-quality?

For the latter questions, it's not essential that you answer "yes" to every one, but you should have a good reason to still use the library after saying "no" to any. Choosing a *good* library can be a tricky business so let's dive into each of these questions to understand how we can choose the best library for our needs.

### Features: Does it provide what I need?

This one should be easy but sometimes isn't. Take a look at documentation, examples, tutorials, read what other developers have said about the library, and maybe take a look at the code itself. You should have an idea what kind of features the library has and how it can help you.

### Software licenses: Can I legally use it?

**CAVEAT**: This is not legal advice. If in doubt, seek your own legal advice (e.g., [UCL Copyright advice][UCL Copy]).

A software license is a way to grant permissions for use and/or distribution to others. Putting your code on a public website does not grant permissions to anyone to use, copy, translate or distribute it as you still own the copyright of that work (even when it's not stated that you own the copyright).

Contrary to popular belief, distributed unlicensed software (not in the public domain) is fully copyright protected, and therefore legally unusable (as no usage rights at all are granted by a license) until it passes into public domain after the [copyright term][cterm] has expired. See [Wikipedia][SL-wiki] for more.

Remember: even if you aren't distributing code yet, you need to understand the licenses of your dependencies.

#### Third party licenses

When you distribute your code, the licenses of any libraries you use takes effect. For example, a library with license:

* [MIT][MITLicense] or [BSD][BSDLicense] are permissive. So you can do what you want, including sell it on.
* [Apache][ApacheLicense] handles multiple contributors and patent rights, but is basically permissive.

Some libraries can affect how you yourself must license your code:

* [GPL][GPLLicense] requires you to open-source your code, including changes to the library you imported, and your work is considered a "derivative work", so must be shared.
* [LGPL][LGPLLicense] for libraries, but [use dynamic not static linkage][LGPLStaticVsDynamic]. If you use static linking its basically as per [GPL][GPLLicense].

However, there's still some debate on [GPL/LGPL and derivative works](https://lwn.net/Articles/548216/). The only true test is in court.

#### Choosing a license

When you plan to distribute code:

* Don't write your own license, unless you use legal advice and you understand its consequences.
* Check [GitHub's advice][Chooselicense], and [OSI][OSI] for choosing your own license.
* Try to pick one of the standard ones for [compatibility][LicesnseCompatibilityWiki].

For an in-depth understanding we recommend you read some works about licenses:

- [Understanding Open Source and Free Software Licensing][LicensingBookWC] ([web][LicensingBook], [UCL][LicensingBookUCL], [pdf][LicensingBookPDF]),
- [The public domain][PDBook]

**Note**: Once a 3rd party has your code under a license agreement, their restrictions are determined by that version of the code.

### Stability: Is a license stable?

Some libraries are so new their *public API* or *interface* is still subject to change. This is usually signalled by the project being in *alpha* or *beta* stages, either before an initial 1.0.0 release, or before a new major x.0.0 release. Some projects (like Python itself) ensure that all *minor* versions will not intentionally introduce breaking changes (i.e. you can use the same code moving from 3.10 to 3.11) but keep *breaking changes* to new major versions (i.e. moving from Python 2 to Python 3). If you haven't come across this idea, read about [semantic versioning](https://www.geeksforgeeks.org/introduction-semantic-versioning/).

When choosing a library to use with your own project, try to use a *stable* version, i.e. one with a stable interface. 

### Efficiency: Is the library fast enough?

Libraries, particularly the good ones, tend to be well-optimised, that is their algorithms and data structures have been tweaked to get the best performance. For performance-critical libraries (like many used in numerical computations) the library developers should include some details about the performance of the library in its documentation. This is where you should ideally look for information about the performance. Otherwise, try to find comparisons with other, similar libraries to understand the performance.

While you are unlikely to beat a library's performance with a custom algorithm, sometimes custom code can be faster due to a tradeoff between flexibility and performance. If you have already used the library but think you might be able to beat a library's performance:

1. test the performance of the library's implementation
2. write some unit tests using the library's implementation
3. write your custom implementation
4. modify the unit tests to test your implementation
5. test the performance of your custom code and compare to the library performance

By testing correctness and performance on both the library and your custom code, you can understand whether it's worth it to commit to either.

### Up-to-date: Is the library regularly updated?

Libraries that are not regularly maintained can "rot", that is:

- bugs don't get fixed
- bugs in dependencies don't get fixed
- new language features break the library
- newer, safer language features don't get introduced
- advances in packaging make it more difficult to install

In general though, we want to avoid these issues, so consider these questions when deciding if the library is suitably up-to-date:

* When was the last release?
* Is there a sensible versioning scheme (e.g. [semantic versioning][semver])?
* Is a [changelog][changelog] provided with each new release?
* Is there a suitable release schedule?
* Is the code developed on the open (e.g., on GitHub)?
  * How often are there commits?

You should develop your own intuition for what you consider "suitably up-to-date" but here are some heuristics of mine:

- If a library has been updated within the last year, it's probably good.
- If a library is very small, it probably doesn't need many updates, so longer releases are fine.
- If a library is very old (like some numerical libraries) then it is so well-used, there probably aren't many bugs left, so a release over ten years ago is still probably okay (but might not be very efficient on modern hardware).

### Ownership: Who develops the library?

Libraries must be developed by someone; if there is no community or company responsible for a library's development, it is considered *abandoned* and should probably be avoided. Consider some of the following questions:

- Is the library obviously developed by a person, community, company or other organisation?
- If a company:
  - How easy is it to report bugs?
  - Is it still open source?
  - What happens if the company decides to make the library closed-source?
- If a person or community:
  - Is the library popular?
  - Are there many contributors to the project?
  - Are issues dealt with sensibly?
  - Is it easy to reach out to the developers?

### Correctness: Is the library well-tested?

* Are there many unit tests, do they run, do they pass?
  * Are they run automatically (i.e. through continuous integration)?
* Does the library depend on other libraries?
* Are the build tools common?

### Quality: Is the library of high quality?

Beyond the things we've already discussed, there are a few more minor points that signal whether a library is developed *well*. The lack of any of these things doesn't mean a library is bad, it may just be more difficult to use or update. Consider:

* Documentation: does it exist? is it good?
* Number of ToDos: do they keep a track of bugs to fix and future features to implement?
* Dependencies: does it offer a clear list of dependencies? Are they trusted? (i.e., recursively)
* Data Structures: is it clear how to import/export data or images to use later?
* Clear API: can you write a convenient wrapper?

## Libraries you should be using

While you should be asking yourself the above questions to understand how a library can help you, there are some groups of libraries you should consider first:

- Standard library
  - *very* well-tested
  - *very* well-documented
  - *very* well-used
  - constantly developed
  - no need to install anything!
- Vendor-provided libraries - provided by Intel/Nvidia/AMD/etc
  - well-tested
  - (often) well-documented
  - usually best performance for a particular architecture
  - but not always open-source
- Well-known libraries - Boost, FFTW, Eigen, Vulkan, etc
  - well-tested
  - well-used
  - (often) well-documented
  - strong communities

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
[LicensingBookUCL]: "https://ucl-new-primo.hosted.exlibrisgroup.com/primo-explore/fulldisplay?docid=UCL_LMS_DS51341932510004761&context=L&vid=UCL_VU2&lang=en_US&search_scope=CSCOP_UCL&adaptor=Local%20Search%20Engine&tab=local&query=any,contains,Understanding%20open%20source%20%26%20free%20software%20licensing&offset=0"
[LicensingBookPDF]: https://people.debian.org/~dktrkranz/legal/Understanding%20Open%20Source%20and%20Free%20Software%20Licensing.pdf
[LicesnseCompatibilityWiki]: https://en.wikipedia.org/wiki/License_compatibility
[PDBook]: https://www.thepublicdomain.org/download/
[changelog]: https://keepachangelog.com/en/1.0.0/
