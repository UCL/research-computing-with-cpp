---
title: Deployment
---

## Deployment

### Getting our code onto Legion

We now have to

* Clone our code base onto our cluster
* Load appropriate modules to get the compilers we need
* Build our code
* Construct an appropriate submission script
* Submit the submission script
* Wait for the job to queue and run
* Copy the data back from the cluster
* Analyse it to display our results.

This is a **pain in the neck**

### Scripting deployment

There are various tools that can be used to automate this process.

You should use one.

Since I like Python, I use [Fabric](http://docs.fabfile.org/en/1.10/index.html)
to do this.

You create `fabfile.py` in your top level folder, and at the shell you can write:

``` bash
fab legion.cold
fab legion.sub
fab legion.stat
fab legion.fetch
```

to build and run code on Legion, without ever using ssh. You can be in any folder
below the `fabfile.py` level, such as inside a build folder; the command line tool
recurses upward to look for this file, (Just like `git` looks for the `.git` folder.)

### Writing fabric tasks

If you know Python, writing fabric tasks is easy:

fabfile.py:

``` python
@task
def build():
    with cd('/home/ucgajhe/smooth/build'):
        with prefix('module load cmake'):
            run('make')
```

``` bash
fab build
```

### Templating jobscripts

Editing a jobscript every time you want to change the number of
cores you want to run on is tedious. I use a templating tool 
[Mako](http://www.makotemplates.org) to
generate the jobscript:

``` mako
#$ -pe openmpi ${processes}
```

the templating tool fills in anything in ${} from a variable in the fabric code.

### Configuration files

Avoid using lots of command line arguments to configure your program.
It's easier to just have one argument, a configuration file path.

* The configuration file can be kept with results for your records.
* Configuration files can be shipped back and forth to the cluster with fabric.

Even if your code is using simple C++ I/O rather than a nice formatting library,
it's best to design your config file in a format which other frameworks can easily read.
My favourite is [Yaml](http://www.yaml.org).

``` yaml
width: 200
height: 100
range: 5
```

### Results

[It works](https://www.youtube.com/watch?v=3sXO2rYNwl4)
