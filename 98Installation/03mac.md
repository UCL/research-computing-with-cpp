---
title: Mac
---


Mac Install
===========

## XCode and command line tools ##

Install [XCode](https://itunes.apple.com/us/app/xcode/id497799835) using the Mac app store.

Then, go to Xcode...Preferences...Downloads... and install the command line tools option

## Git ##

Install Homebrew via typing this at a terminal:

``` Bash
ruby -e "$(curl -fsSL https://raw.github.com/mxcl/homebrew/go)"
```    

and then type

``` Bash
brew install git
```


Then install the [GitHub for Mac client](http://mac.github.com). (If you have problems with older versions of OSX, it's safe to skip this.)

## CMake

Just do

``` Bash
brew install cmake
```

Minimum version 3.5.

## Editor and shell ##

The default text editor on OS X *textedit* should be sufficient for our use. Alternatively
choose from a [list](http://mac.appstorm.net/roundups/office-roundups/top-10-mac-text-editors/) of other good editors.

To setup git to use *textedit* executing the following in a terminal should do.

``` Bash
git config --global core.editor
	/Applications/TextEdit.app/Contents/MacOS/TextEdit
```

The default terminal on OSX should also be sufficient. If you want a more advanced terminal [iTerm2](http://www.iterm2.com/) is an alternative.
