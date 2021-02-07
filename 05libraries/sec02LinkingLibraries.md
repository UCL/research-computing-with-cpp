---
title: Linking Libraries
---

## Linking libraries

From the [first lecture][lesson-first] we've seen that:
    * Code is split into functions/classes
    * Related functions get grouped into libraries
    * Libraries get compiled / archived into one file

And that the End User of a library needs access to:
    * Header file = declarations (and implementation if header only)
    * Object code / library file = implementations

The pre-compiled libraries usually can be added to our projects via two mechanism, i.e., via
static or dynamic linking. Let's see their differences when using one or the other:
    
### Static Linking

* Their extension are `.lib` for Windows and `.a` for Mac or Linux.
* Compiled code from static library is **copied** into the current translation unit while is built.
* Therefore, it uses more disk space compared with dynamic linking.
* Current translation unit then does not depend on that library (i.e., only have
  to distribute the executable to run your program).

Find how to create static libraries on the following videos:
- Cave of Programming's C++ Tutorial [Static Creating Libraries][CPPCoPStatic] (MacOS + Eclipse)
- The Cherno's C++ series [Using Libraries in C++ (Static Linking)][CPPChernoStacic] (Windows + VS2017)
- iFoucs's [How to create a static library][ProgLinIF_static] (Linux + CLI)


### Dynamic Linking

* Windows (.dll), Mac (.dylib), Linux (.so)
* Compiled code is left in the library.
* At **runtime**,
    * OS loads the executable
    * OS / Linker finds any unresolved libraries in the system
    * Recursive process
* Saves disk space compared with static linking.
* Faster compilation/linking times?
* Current translation unit has a known dependency remaining.
* The libraries can be updated to a newer version without requiring
  recompilation of the executables that uses it (if the interfaces haven't
  changed).
    
You can see some examples on how to create dynamic libraries at:
- iFoucs's [How to use a Dynamic Library][ProgLinIF_dyanmic] (Linux + CLI)
- The Cherno's C++ series [Using Dynamic Libraries in C++][CPPChernoDynamic] (Windows + VS2017)
    
### Dynamic Loading

[Dynamic loading][DynamicLoading-wiki] is a third mechanism that we are not
covering here. It's normally used for plugins.

To load the libraries you need using system calls with `dlopen` (*nix) or
`LoadLibrary` (Windows). This allows for dynamically discovering function names
and variables.


## Linking in practice

Though you can do all the linking manually as seen in the [previous
page][lesson-LibBasics], as your project grows it's better to use some tool that
automate the process for you. Check this short video about [how to add a library using CMake][CPPVoBCMakeAddLib].


### Space Comparison

If you have many executables linking a common set of libraries:
* Static
    * Code gets copied - each executable becomes bigger
    * Doesn't require searching for libraries at run-time
* Dynamic
    * Code gets referenced - smaller executable
    * Requires finding libraries at run-time

However, space is less of a concern these days!


### For Scientists

As a scientists, we want that our programs are:
* Easy to use
* Easy to distribute to collaborators
Therefore, we tend to prefer static if possible for ease of deployment.


### Packaging

C++ doesn't have yet a standard way to distribute and package libraries (as Python
or Rust), but there are many options available depending of your userbase.
Note that packaging large apps takes effort!

Some packaging systems available are:
* OS package managers ([`apt`][DebPack], [`brew`][BrewPack], [`chocolatey`][ChocoPack])
* CMake repositories ([Hunter][HunterPack], [CPM][CPM])
* C++ package managers ([conan][ConanPack], [vcpkg][vcpkg], [buckaroo][buckPack])
* General package managers ([spack][SpackPack], [easybuild][ebPack], [conda][condaPack])

Packaging is not easy, but there are some [things you can do to make package managers cry][make-PM-cry] 😭


### Checking the dependencies

These tools will help you to see whether your executables require any shared library:

* Windows - Dependency Walker
* Linux - `ldd`
* Mac - `otool -L`




[lesson-first]: ../01research/
[CPPCoPStatic]: https://www.youtube.com/watch?v=kw3UD_YCoEk
[CPPChernoStacic]: https://www.youtube.com/watch?v=or1dAmUO8k0
[ProgLinIF_static]: https://www.youtube.com/watch?v=3RmIVDgPmGk
[DynamicLoading-wiki]: https://en.wikipedia.org/wiki/Dynamic_loading
[ProgLinIF_dyanmic]: https://www.youtube.com/watch?v=pkMg_df8gHs
[CPPChernoDynamic]: https://www.youtube.com/watch?v=pLy69V2F_8M
[lesson-LibBasics]: ./sec02LibraryBasics.html
[CPPVoBCMakeAddLib]: https://www.youtube.com/watch?v=abuCXC3t6eQ
[DebPack]: https://wiki.debian.org/HowToPackageForDebian
[BrewPack]: https://docs.brew.sh/Formula-Cookbook
[ChocoPack]: https://docs.chocolatey.org/en-us/features/create-packages
[HunterPack]: https://hunter.readthedocs.io/en/latest/creating-new.html
[CPM]: https://github.com/TheLartians/CPM.cmake
[ConanPack]: https://docs.conan.io/en/latest/creating_packages.html
[vcpkgPack]: https://docs.microsoft.com/en-us/cpp/build/vcpkg
[buckPack]: https://github.com/LoopPerfect/buckaroo/wiki/Creating-a-Package
[SpackPack]: https://spack.readthedocs.io/en/latest/packaging_guide.html
[ebPack]: https://docs.easybuild.io/en/latest/Writing_easyconfig_files.html
[condaPack]: https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs.html
[make-PM-cry]: https://www.youtube.com/watch?v=NSemlYagjIU
