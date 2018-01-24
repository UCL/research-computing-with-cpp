---
title: Linux
---

Linux Install
=============

Git
---

If git is not already available on your machine
you can try to install it via your distribution package manager (e.g. `apt-get` or `yum`).

On ubuntu or other Debian systems:

    sudo apt-get install git

On RedHat based systems:

    sudo yum install git

## CMake ##

Again, install the appropriate package with apt-get or yum (`cmake`). Minimum version 3.5.

## Editor and shell ##

Many different text editors suitable for programming are available.
If you don't already have a favourite, you could look at [Kate](http://kate-editor.org/).

Regardless of which editor you have chosen you should configure git to use it. Executing something like this in a terminal should work:

```
git config --global core.editor NameofYourEditorHere
```

The default shell is usually bash but if not you can get to bash by opening a terminal and typing `bash`.
