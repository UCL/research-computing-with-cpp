---
title: Terminal Commands Cheat Sheet
---

# Terminal Commands Cheat Sheet

You will need to use the terminal on your machine every week for cloning code repositories, compiling code, and running executables. This page provides some of the most common commands that you will need to use in your terminal. Where you see angle brackets such as `<directory>` this means that this is an argument for a command, and should be replaced (in this example, with the name of a directory). 

## Basic Terminal Commands

- `pwd`: This displays the _present working directory_. This is where you currently are in the filesystem.
- `cd <directory>`: This is _change directory_, and allows you to move to a different folder. Use `cd ..` to move up one level to the directory that contains the present working directory. You can go up multiple levels by `cd ../..` and so on. 
- `ls`: This displays the contents of the current directory. You can write `ls <directory>` to display the contents of a directory that you are not in. If you write `ls -la` instead of just `ls` you will get more information, like when the file was created.
- `mkdir <name>`: Creates a new directory with the given name. 
- `cp <source> <destination>`: Copies a source file to a new destination. If the destination is a directory then it will copy the file into that directory and keep the name; if the destination includes a file name then the copy of the file will be given the new name.
- `mv <source> <destination>`: The same as `cp` but it _moves_ the file instead of copying it. 
- `rm <file>`: Remove (i.e. delete) a file.
- `rm -rf <directory>`: Remove an entire directory.
- `code <directory>`: Open a folder with VSCode. Use `code .` to open the current directory in VSCode. 

## Git 

- `git clone <repo>`: Clone a git repo.
- `git add <file>`: Add a file to the current commit.
- `git commit -m <message>`: Finalise a git commit with a given message.
- `git checkout -b <branch_name>`: Create a new git branch.
- `git push`: Push to the repository on the current branch.
    - use `git push --set-upstream origin <branch_name>` if this is the first time using a new branch.
- `git rm`: Delete a file and remove it from git tracking.
- `git mv`: Move a file and update git tracking.

## Compiling and Running Programs

These commands are a quick reference for a number of tools that will be introduced over this course. They will be discussed more fully in their respective sections, so do not worry if you do not know what these are and do not expect to be able to use these right away. However, you may find it helpful to come back here to quickly look up useful commands. 

- `g++ -o <output> -std=c++17 -I<include_directory> <sources>`: Compile a C++ program using the C++17 standard using `g++`. Use `-g` as well when you want to compile with debug symbols. 
- `cmake -B <build_folder>`: Create a build folder and initialise cmake therein.
- `cmake --build <build_folder>`: Build a CMake project in the build folder.
- `./<path_to_executable>`: Run an executable. If the executable takes any arguments then place them after this, e.g. `./my_exe 5 10`.  
- `gdb <path_to_exe>`: Run an executable in command line debugger (Ubuntu/WSL).
- `valgrind ./<path_to_exe>`: Run an executable with valgrind. 
- `g++ ... -fopenmp`: Compile with OpenMP turned on. (Other arguments omitted for brevity.)
- `OMP_NUM_THREADS=<n_threads> ./<path_to_executable>`: Run an OpenMP program with a given number of threads. (Arguments can be supplied after the exe.)
- `mpic++ -o <output> -std=c++17 -I<include_directory> <sources>`: Compile a C++ MPI program.
- `mpirun -np <n_proc> <path_to_exe>`: Run an MPI program with a given number of processes. (Arguments can be supplied after the exe.)