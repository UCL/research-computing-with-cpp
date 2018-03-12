#!/bin/bash -l
#$ -S /bin/bash
#$ -P RCSoftDev
#$ -l h_rt=3:0:0
#$ -N SmoothLife
#$ -pe openmpi 16
#$ -wd /home/ucgajhe/Scratch/Smooth/output
module unload compilers
module unload mpi
module load compilers/gnu/4.6.3
module load mpi/openmpi/1.6.5/gnu.4.6.3
gerun /home/ucgajhe/devel/smooth/SmoothLifeExample/build/src/smooth_mpi config.yml
