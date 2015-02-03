---
title: Job Scheduling
---

## Batch Computing

### High Throughput Computing

* High Throughput Computing (HTC)
* See [RITS][RITS]/[RSDT][RSDT] course "[HPC and HTC using Legion][HPTCLegion]"
* Also, see [Unix Shell Tutorial][UnixShell]


### What is it?

* Taking same job/process (includes script containing sequence of commands)
* Applying repeatedly to same data
* SPMD concept


### What advantages?

* Massive throughput
* In imaging
    * Freesurfer job takes 24h
    * ADNI dataset, > 1000 images
    * Don't want to wait 1000 days (about 3 years)
    * Run all in parallel
    * Depends on number of nodes available
* No change to code as such
    * Need to compile on cluster 
    * Maybe set LD_LIBRARY_PATH at runtime
    
### What disadvantages?

* Clusters normally Linux
    * (Disadvantage if you are not using Linux)


### How Do I Start?

1. Get Legion Account
1. Make sure you know [Unix Shell][UnixShell]
1. Read [Legion Course][HPTCLegion]
1. Read [Legion Wiki][LegionWiki]
1. Refer to [Legion Cheat Sheet][LegionCheats]
1. Write [submission script][LegionScript]
1. Run Jobs!


### Example - Calculate Pi

(Same example as on [Legion course][HPTCLegion])

* Log into Legion.
* Example simply runs a program to calculate pi.

```
cd ~/Scratch
cp -r /shared/ucl/apps/examples/calculate_pi_dir ./
cd calculate_pi_dir
make
./calculate_pi
```

(we can run standalone, but don't run huge jobs like this, see [policies][LegionPolicies])


### Submission Script

In file submit.sh

```
#!/bin/bash -l
#$ -S /bin/bash
#$ -l h_rt=0:10:00
#$ -l mem=1M
#$ -N <job name>
#$ -P <project name>
#$ -wd /home/skgtmjc/Scratch/calculate_pi_dir

./calculate_pi
```
replacing things between angled brackets.


### Run via qsub

```
qsub submit.sh
```


### Basic qsub commands

```
qsub <script>  # submit to queue according to scheduler options
qstat          # show status of queued jobs
qdel <pid>     # delete my job with ID <pid>
```

See also [Legion cheat sheet][LegionCheats], and [example submission scripts][LegionScript].


### Run multiple jobs

* For repeated use:
    * e.g. same program, many different images
    * run qsub with script name and different arguments, processing arguments in script

```
echo "Ive been called with image $1"
./some_process $1
```

and

```
qsub submit.sh image1.nii
qsub submit.sh image2.nii
```
(or put in a loop)
    
[RITS]: http://www.ucl.ac.uk/research-it-services/homepage
[RSDT]: http://www.ucl.ac.uk/research-it-services/our-work/research-software-development
[HPTCLegion]: http://development.rc.ucl.ac.uk/training/hptclegion/
[UnixShell]: http://development.rc.ucl.ac.uk/training/hptclegion/SWCShell/
[LegionWiki]: https://wiki.rc.ucl.ac.uk/wiki/Legion
[LegionScript]: https://wiki.rc.ucl.ac.uk/wiki/Example_Submission_Scripts_for_Legion
[LegionCheats]: https://wiki.rc.ucl.ac.uk/mediawiki119/images/a/ad/Legion_ref_sheet.pdf
[LegionPolicies]: https://wiki.rc.ucl.ac.uk/wiki/Legion_Resource_Allocation