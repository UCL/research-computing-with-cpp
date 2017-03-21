---
title: Tools for pretty code
---

## The end of brace wars: brainless auto formatting

Code is read more often than written:

  ``` cpp
int GCD(int a,int b)
{int r;while(b){r=a%b;
  a=b;b=r;}return a;}
  ```

  [clang-format](https://clang.llvm.org/docs/ClangFormat.html) is one possible
  code formatter. Add it (or any equivalent) to you editor for automatic
  formating.

## Linting

  Software to check code for correctness:

  - the compilers themselves: "-Wall"
  - [clang-tidy](http://clang.llvm.org/extra/clang-tidy/)
  - [cppcheck](http://cppcheck.sourceforge.net/)

  example:

  ``` cpp
  if(FFTW_plan_flag != FFTW_ESTIMATE | FFTW_PRESERVE_INPUT) {
    ...
  }
```

## Refactoring

Once tests exist, it is easy and safe to modify the code:

Refactoring means rewriting:

- to simplify existing code
- to simplify future development
- for legibility
- to decrease tech-debt
- to consolidate similar code (avoid copy-pasta)

