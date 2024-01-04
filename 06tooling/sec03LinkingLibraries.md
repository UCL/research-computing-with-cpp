---
title: Linking Libraries
---

## Linking libraries

So far in the course we've seen that:

* Code is split into functions/classes
* Related functions get grouped into libraries
* Libraries get compiled / archived into one file

And that the End User of a library needs access to:

* Header files
  - These contain required declarations 
  - May also include implementation if the library is _header only_. This is relatively common for heavily templated code that must be so general that the templates cannot be explicitly instantiated in source files in the library.
* Object code / library file
  - This is usually where the implementation is found.

The pre-compiled libraries usually can be added to our projects via two mechanism, i.e., via
static or dynamic linking. Let's see their differences when using one or the other:
    
### Static Linking

* Their extension are `.lib` for Windows and `.a` for Mac or Linux.
* Compiled code from static library is **copied** into the current translation unit while is built.
* Therefore, it uses more disk space compared with dynamic linking.
* Current translation unit then does not depend on that library (i.e., only have
  to distribute the executable to run your program).

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
    
### Dynamic Loading

[Dynamic loading][DynamicLoading-wiki] is a third mechanism that we are not
covering here. It's normally used for plugins.

To load the libraries you need using system calls with `dlopen` (Linux or Mac) or
`LoadLibrary` (Windows). This allows for dynamically discovering function names
and variables.


## Linking in practice

Though you can do all the linking manually as seen in the [previous
page][lesson-LibBasics], as your project grows it's better to use a build tool like CMake.

We'll explore building static and shared libraries in the exercises in class, but here are some key pointers:
- The [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html) in their official docs has chapters on creating libraries as part of your CMake projects. This is a good starting point for creating an internal library. 
  - You can switch declare a library `STATIC` or `SHARED` when you add it
- You can compile a library without creating an executable if you want a library to be a separate project: simply use the [`add_library`](https://cmake.org/cmake/help/latest/command/add_library.html) command! You don't need to have an `add_executable`. 
- You should probably set the `CMAKE_LIBRARY_OUTPUT_DIRECTORY` variable in your top level CMake so that your compiled library is easy to find, similar to how our executables are placed in `CMAKE_RUNTIME_OUTPUT_DIRECTORY`.  
- You can also [import a library into a project](https://cmake.org/cmake/help/latest/command/add_library.html#imported-libraries). If you have compiled your library as a separate project and you want to use it in an executable, you'll need to import it. 
  - You'll want to set the [`IMPORTED_LOCATION`](https://cmake.org/cmake/help/latest/prop_tgt/IMPORTED_LOCATION.html) property in `set_target_properties` to the location of your compiled (shared or static) library file. 
  - Make sure to set the executable's include path so it can find the headers for your imported library! 
- In practice we often use CMake's [find_package](https://cmake.org/cmake/help/latest/command/find_package.html) command to find libraries that are installed in standard locations, so that the same CMake will work on multiple people's computers who may have libraries in slightly different locations. This saves them having to edit their CMake files to provide an exact location unless necessary. 

### Space Comparison

If you have many executables linking a common set of libraries:

* Static
    * Code gets copied - each executable becomes bigger
    * Doesn't require searching for libraries at run-time
    * Code needs to be re-compiled to reflect changes in libraries
    * Executables are independent of one another even if they use the same libraries.
* Dynamic
    * Code gets referenced - smaller executable
    * Requires finding libraries at run-time
    * Library code can be updated without re-compiling executables
    * Executables which use the same shared library object are all affected by changes to that library, so you need to avoid breaking any interfaces or functionality that other programs rely on. 


### Packaging

C++ doesn't have yet a standard way to distribute and package libraries (as Python
or Rust), but there are many options available depending of your userbase.
Note that packaging large apps takes effort!

Some packaging systems available are:

* OS package managers ([`apt`][DebPack], [`brew`][BrewPack], [`chocolatey`][ChocoPack])
* CMake repositories ([Hunter][HunterPack], [CPM][CPM])
* C++ package managers ([conan][ConanPack], [vcpkg][vcpkg], [buckaroo][buckPack])
* General package managers ([spack][SpackPack], [easybuild][ebPack], [conda][condaPack])

Packaging is not easy, but here is a humorous and enlightening talk about [things you can do to make package managers cry][make-PM-cry]. 


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
[vcpkg]: https://docs.microsoft.com/en-us/cpp/build/vcpkg
[buckPack]: https://github.com/LoopPerfect/buckaroo/wiki/Creating-a-Package
[SpackPack]: https://spack.readthedocs.io/en/latest/packaging_guide.html
[ebPack]: https://docs.easybuild.io/en/latest/Writing_easyconfig_files.html
[condaPack]: https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs.html
[make-PM-cry]: https://www.youtube.com/watch?v=NSemlYagjIU
