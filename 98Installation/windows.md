---
title: Windows
---

Windows Install
===============

## Git ##

Install [msysgit](http://code.google.com/p/msysgit/downloads/list?q=full+installer+official+git)

Then install the [GitHub for Windows client](http://windows.github.com/).

## Subversion

Install [subversion](http://sourceforge.net/projects/win32svn/)

And choose to add it to the path for all users if so prompted.

## CMake

Install [cmake](http://www.cmake.org/cmake/resources/software.html)

And choose to add it to the path for all users if so prompted.

## Editor ##

Unless you already use a specific editor which you are comfortable with we recommend using
[*Notepad++*](http://notepad-plus-plus.org/) on windows.

Using Notepad++ to edit text files including code should be straight forward but in addition you should configure git
to use notepad++ when writing commit messages (We will learn about these in the version controle session).   

## Unix tools ##

Install [MinGW](http://sourceforge.net/projects/mingw/) by following the download link.
It should install MinGW's package manager. On the left, select ``Basic Setup``, and on the right select
``mingw32-base``, ``mingw-developer-toolkit,``
``mingw-gcc-g++`` and ``msys-base``. On some systems these package
might be selected from start. Finally, click the installation menu and ``Apply Changes``.

## Locating your install

Now, we need to find out where Git and Notepad++ have been installed, this will be either in
`C:/Program Files (x86)` or in `C:\ProgramFiles`. The former is the norm on more modern versions of windows.
If you have the older version, replace `Program\ Files\ \(x86\)` with `Program\ Files` in the instructions below.

## Telling Shell where to find the tools

We need to tell the new shell installed in this way where git and Notepad++ are.

To do this, use NotePad++ to edit the file at `C:\MinGW\mysys\1.0\etc\profile`

and toward the end, above the line `alias clear=clsb` add the following:

``` Bash
# Path settings from SoftwareCarpentry
export PATH=$PATH:/c/Program\ Files\ \(x86\)/Git/bin
export PATH=$PATH:/c/Program\ Files\ \(x86\)/Notepad++
# End of Software carpentry settings
```


## Finding your terminal

Check this works by opening MinGW shell, with the start menu (Start->All programs->MinGW->MinGW
Shell). This should open a *terminal* window, where commands can be typed in directly.

On windows 8,
there may be no app for MinGW. In that case, open the ``run`` app and type in

``` Bash
C:\MinGW\msys\1.0\msys.bat
```

## Checking which tools you have

Once you have a terminal open, type

``` Bash
which notepad++
```

which should produce readout similar to

```
/c/Program Files (x86)/Notepad++/notepad++.exe`
```

Also try:

``` Bash
which git
```

which should produce

```
/c/Program Files (x86)/Notepad++/notepad++.exe
```

The ``which`` command is used to figure out where a given program is located on disk.

## Tell Git about your editor

Now we need to update the default editor used by Git.

``` Bash
git config --global core.editor "'C:/Program Files (x86)/Notepad++
	/notepad++.exe' -multiInst  -nosession -noPlugin"
```

Note that it is not obvious how to copy and paste text in a Windows terminal including Git Bash.
Copy and paste can be found by right clicking on the top bar of the window and selecting the
commands from the drop down menu (in a sub menu).  

You should now have a working version of git and notepad++, accessible from your shell.
