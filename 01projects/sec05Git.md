---
title: Version control with Git
---

Estimated reading time: 35 minutes

## Introduction

Professional software projects employ version control systems for detailed tracking of a project's history. Version control systems is a large topic. In this short introduction, we cover just the bare minimum essentials on using a popular version control system known as `git`.

## Installing and configuring `git`

### Installation

Git is a cross-platform software, and can be installed on most major operating systems and processor architectures. One method to install Git is from the project's [official website](https://git-scm.com/). Git has already been pre-installed within the official [devcontainer image](https://github.com/UCL/rc-cpp-vscode-dev-container) provided to you for this class. Please follow the Moodle page on how to set up VS Code with the provided development container (devcontainer).

For any reason, if you are not using the provided VSCode devcontainer image, please attempt to install and configure `git` for your preferred development environment before attending the in-person session for Week 1. In case of difficulties, instructors and teaching assistants are available in the class, to help set this up.

The following command verifies that Git has been successfully installed in your development environment (note that the `$` sign merely represents the shell prompt, which could be different in your environment, and should not be typed in any of the snippets from these notes).

``` sh
$ git --version
```

The above command should display `git version 2.xx.y` (2.39.0 for the official devcontainer for this class), where `xx` and `y` represents the minor and patch revisions that has been installed. Note that the major revision must be `2`, since `git 1.x` is too antiquated at this point and is not recommended anymore.

### Configuration

Prior to using Git for version control, it requires to be set up with basic user configuration. At a minimum, the following two commands need to be run.

``` sh
$ git config --global user.name "Full name"
$ git config --global user.email "email address"
```

In this class, we shall use VSCode as our official editor for writing meta-information about changes to the tracked files in our project, we set it up as follows:

``` sh
$ git config --global core.editor "code --wait"
```

You may choose to use another text editor, and the instructions for configuring some popular editors are given [here](https://swcarpentry.github.io/git-novice/02-setup/index.html).

## Git commands (Exercise 1)

Git is a command-line application, wherein the commands all start with the keyword `git` followed by a sub-command and zero or more options and arguments. The sub-command usually denotes the action to perform. In fact, we just configured Git using the `git config` command in the section above!

Here, we briefly describe a few basic Git commands to help you get started with version controlling a typical software project.

### `git init`

Let us say we are ready to start version controlling the files in our project which is structured as following:

``` sh
./my_app/
├── hello.cpp
├── LICENSE.txt
└── README.md
```

The files listed in this example project hierarchy are available to download as a compressed archive [here](https://liveuclac-my.sharepoint.com/:u:/g/personal/uccagop_ucl_ac_uk/ESoVCHLdn4ZKiQEVxE_vVTwBqP9r6aOgcusKsPii7nN50w?e=wyyt8M). This shall serve as the starting point for the first in-class exercise on `git`. We first initialise Git on the project by moving into the `my_app` folder (i.e. the project's root directory), and typing:

``` sh
$ git init
```
This will create a hidden `.git` folder at the project root (the contents of which is managed by Git and should never be edited by hand). You may verify the presence of this hidden `.git` folder by passing the `-a` (short for `all`) option to the `ls` command as follows:

``` sh
$ ls -a
```

### `git status`

We shall frequently employ the `git status` command to check the tracked status of project files. At this stage, running `git status` returns:

``` sh
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
    LICENSE.txt
    README.md
    hello.cpp
```

As seen in the above output, it is clear that there are untracked files present in the project. There is a helpful hint suggesting the use of `git add <file>` to track files, and we shall look at this next.

### `git add`

Once we are ready to start tracking files, the first step is to add them to Git's internal staging area.  This is done with the `git add` command to which we pass in the list of files as arguments.

``` sh
$ git add hello.cpp
```
will add the `hello.cpp` file to Git's staging index. Running `git status` will now return:

``` sh
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
    new file:   hello.cpp

Untracked files:
  (use "git add <file>..." to include in what will be committed)
    LICENSE.txt
    README.md

```

This indicates that the file `hello.cpp` is ready to be committed. A 'commit' is a permanent version of record that is maintained by Git, and can be referenced in the future by other Git commands.

Shell wildcards may be used for easily adding multiple files with a similar pattern, e.g. `git add *.cpp` shall add all the C++ source files from the current folder, and `git add *.md` will add all the markdown files from the current folder, to the staging area. We may also pass file paths, e.g. `git add <dir_name>/` shall add all the files within the directory `<dir_name>`. A common pattern is to use at the project root:

``` sh
$ git add .
```
where `.` is the reference to the current working directory. When executed at the project root, it adds all files under the project to the staging index. After running `git add .` for our example folder, `git status` returns:

``` sh
On branch main
No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
    new file:   LICENSE.txt
    new file:   README.md
    new file:   hello.cpp
```

Now, all three files are ready to be committed (i.e. made a permanently referenceable entity of the project), and we employ the `git commit` command for this.

### `git commit`

Now, invoking `git commit` on the command line opens up a text editor (configured in the earlier section), and shall wait for a *commit message* from the user. A good commit message shall describe the changes that have happened in this particular commit. By convention, the very first commit for a project usually has the commit message string "Initial commit", and we shall do the same here. After this, the very first version of record (i.e. a commit) has been created.

Invoking `git status` now produces the output:

```
On branch main
nothing to commit, working tree clean
```
and we have a clean working directory with no further git-related actions needed at this point.

## Making changes to tracked files and viewing differences

Let us now edit `README.md` to add a new line of text (with the content of your choice) at the bottom of this file. Running `git status` now results in the following output

```
On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
    modified:   README.md

no changes added to commit (use "git add" and/or "git commit -a")
```

Git is now tracking changes to our README.md, and detects that it has been modified! To view the actual changes to the file, we may use the `git diff` command discussed next.

### `git diff`

In its most basic form of invocation, `git diff` presents a list of changes between the file(s) in the current working directory and the version of record in the most recent commit. Running `git diff` here results in

```
diff --git i/README.md w/README.md
index 909b915..7c3dae9 100644
--- i/README.md
+++ w/README.md
@@ -1 +1,2 @@
 This project contains a simple "Hello World" C++ source code, which when compiled and run, prints "Hello, World!" to the cons
ole.
+This file serves as an example to demonstrate version tracking by git
```

Ignoring the first few lines of the command's output, the changeset information is presented in Unix's unified diff format, wherein the `+` sign indicates that a new line has been added. [This](https://unix.stackexchange.com/a/216131) StackOverflow answer provides more details about Unix's unified diff output syntax.

The usual procedure to commit this change to the repository is through the `git add <filename>` followed by `git commit` commands in sequence. However, adding and committing is so frequently used that Git provides a convenient  shortcut for this with the `-am` flag to `git commit`,  where the `-a` flag stands for `add`, and the `-m` flag indicates that the commit message shall be supplied right at the command-line surrounded by double quotes, rather than having to invoke a text editor application. An example usage is as follows:

```
$ git commit -am "Adds an explantory line to README.md"
```

Next, we shall view the project's history with the `git log` command.


### `git log`

The `git log` command shows the project's commit history. Invoking `git log` all by itself provides a detailed log of the commit history of the project.

The detailed view presented by `git log` consists of each commit's unique identifier (a 40 character unique hexadecimal string known as the commit SHA), followed by author and date information, followed by the commit message. For a large project with many changes, sometimes a quick glance at the project's history is helpful wherein the command-line option `--oneline` comes handy.

Executing `git log --oneline` presents a compact view of the project's history where each commit occupies just one line consisting of just the shortened SHA string and the first line of the commit message.

*End of exercise 1*

## Remote repositories and Github (Exercise 2)

So far, everything we have done has been local i.e. pertains to project files on the user's computer. The project history is tracked by Git using the contents of the `.git` folder. However, for ease of collaboration as well as for data resilience, it is common practice to have the project tracked in a remote location on the cloud.

Several services are available that can remotely host Git repositories. Examples include GitHub, Bitbucket, and Gitlab. In this class, we shall be using Github as the central location of our remote repository.

Firstly, we shall navigate to GitHub's landing page on our browser, create an account, and login to it. Then we shall create a new public repository and assign a name to it. The name of the project on GitHub can be different to the Git folder on the local machine, although it is conventional to use the same name. During this process, it is important to choose `ssh` as the remote connection protocol (the other choice usually presented is `https`, which has been deprecated for a while on GitHub). GitHub shall provide instructions to add the newly created local repository to our local Git repository, which resembles the following:

``` sh
$ git remote add <local_reference_name_for_remote_repo> <remote_url>
```

The `<local_reference_name_for_remote_repo>` is the name that the local Git instance shall use for referring to the remote repository (instead of the full URL) each time a remote operation needs to be performed. It is conventional to use the name `origin` for a remote repository.

Since we are using the SSH connection protocol, the `<remote_url>` shall be of the form `git@github.com:<github_username>/<github_repo_name>.git`

Thus the above command is typically written as:

``` sh
$ git remote add origin git@github.com:<github_username>/<github_repo_name>.git
```

At this stage, the remote repository has been connected to our local repository and we can sync our local commits with the remote. However, to be able to do so requires configuring our computer to make secure connections to GitHub.

### Configuring secure remote connections with SSH 

Note that the below steps may also be performed on the host system (outside the development container) since the devcontainer is able to access the ssh configuration of the host. Teaching assistants shall be available at hand to help with setting up SSH access.

Firstly, we need to authenticate ourselves with GitHub through an SSH keypair handshake mechanism. An SSH keypair consists of two separate keys - 1) a *private* key, and 2) a *public* key. Firstly, we generate a keypair on our local machine by running the `ssh-keygen` command. This command is provided by the OpenSSH client software, and can be installed on Debian/Ubuntu operating systems with the command `sudo apt install openssh-client` (this has already been pre-installed on the official devcontainer provided).

First, we run `ssh-keygen` command. The default prompt answers may be accepted. The keypair will be generated, and shall be located in the `/home/username/.ssh/` folder for regular users, and under `/root/.ssh/` in the official development container. The private key shall be named `id_rsa` by default (note the absence of any file extension), and the public key shall be named `id_rsa.pub` (i.e. has the same name as the private key followed by a `.pub` extension). It should be noted that the private key should not be shared with anyone else (including your instructors!), but the public key may be freely distributed anywhere.

Next, open the public key file in a text editor, and copy its entire contents to the system clipboard. On GitHub login to your user account, and under user settings, set up SSH key access to repositories by pasting the public key string from clipboard. 

Your local laptop/devcontainer is now ready to synchronise any local git repository with its remote counterpart on github and the commands to do that are discussed next.

### `git push` and `git pull`

At this stage, we are ready to push our local git history to the remote repository that we created earlier. To do this, run the following command:

```
$ git push -u origin main
```

The `git push` command facilities pushing of local changes to the remote. However, we need to specify the local reference name for the remote repository (i.e. `origin` by convention as we set it up earlier) and the current branch (`main` in newer versions of Git). Note that we do not cover Git branches in depth in this introductory lesson. The `-u` flag indicates that the specified local branch (`main`) shall always be considered to correspond to the same branch in the specified remote. From now on, further commits in the local repository can be pushed by simply invoking `git push` without additional arguments.

If we work on another computer and push commits to GitHub (or a collaborator pushes commits to GitHub), it is required to first pull down and merge these remote updates to our local repository before we can push. The simplest command to achieve this is `git pull` (provided that the current branch has been set to correspond to its remote counterpart as explained above).

Note that it is important to `git pull` as frequently as possible to minimise the possibilities of merge conflicts. The topic of merge conflicts is not covered in this introductory lesson, but is quite important to understand in professional software development.

## Accessing public open source repositories on local machine without SSH

### `git clone`

It should be mentioned that setting up SSH access is required only for pushing commits to a remote repository. However, if we are merely interested in pulling down the complete project history of a public remote repository, a standard HTTPS access without any special setup shall suffice. The first 'download' of the project files along with its revision history is called a *clone* and can be achieved with the `git clone` command, whose typical invocation is as follows:

```
$ git clone https://github.com/<username>/<repo_name>.git
```

This works only for publicly accessible repositories hosted on remote hosting platforms like GitHub. The above command shall fetch the current version of the project files along with its complete git history into a folder titled `repo_name` on the user's local computer as a standard local repository. Then onwards, further commits to the remote repository can be obtained by using the standard `git pull` command.

### Git Submodules

Projects tracked in git repositories could depend on logically independent third party files, which themselves might be under version control. For example, a medical image processing software for a particular application area could depend on a custom domain-specific file compression library that is co-developed in the same project and is reasonable to keep the library's files in a subfolder within the main project tree. Since the file compression library itself is logically distinct from the application within which it is embedded into, it is meaningful to track versions of the library as a separate Git project from the main application (which itself is version controlled in Git). Yet another example is where our application code depends on a third party project which is not located within the project tree, but rather is tracked separately as an independently developed project. In both these scenarios, we make use of Git *submodules*.

The path to the submodule (either a local folder or a remote repository) is configured in a `.gitmodules` file at the root of the main project tree (i.e. the 'outer' repository), and the relevant submodule code is brought in through the `--recurse-submodules` flag to commands like `git clone` and `git pull`.

As an example for cloning a remote repository along with relevant submodule files, we invoke:

```
$ git clone --recurse-submodules https://github.com/<username>/<remote_repo>.git
```

In fact, this is how we shall proceed with the devcontainer setup for our upcoming CMake exercises. The devcontainer configuration is maintained in a separate repository which has been configured in a relevant `.gitmodules` file in our CMake exercise repository.

## Further resources

We have only covered a very basic overview of the Git version control system that shall enable us to get started with the in-class exercises and course projects. An excellent resource that provides an expanded introduction is the Software Carpentry's [lessons on Git](https://swcarpentry.github.io/git-novice/) which covers some additional topics such as ignoring certain kind of files from being tracked, referencing previous commits in git commands etc. The software carpentry lesson material has been taught as a video playlist with live coding/demonstrator by your course instructor and is available [here](https://www.youtube.com/playlist?list=PLn8I4rGvUPf6qxv2KRN_wK7inXHJH6AIJ).

In professional software development, one usually encounters further advanced topics such as branching, rebasing, cherry-picking commits etc for which specialised git resources exist both online and in print. All UCL students have free access to content from LinkedIn Learning, and it is worthwhile to look into some of the [top rated Git courses](https://www.linkedin.com/learning/search?keywords=git&upsellOrderOrigin=default_guest_learning&sortBy=RELEVANCE&entityType=COURSE&softwareNames=Git) there.
