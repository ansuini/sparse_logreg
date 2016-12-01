#!/bin/bash
#PBS -N prof_gridsearch
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=20
#PBS -T flush_cache
#PBS -q regular

NAME=prof_gridsearch
WD='/scratch/ansuini/repositories/machine_learning/sparse_logreg/code'
cd $WD
echo 'Working directory is' $WD
rm -f prof_gridsearch*


export PATH="/home/ansuini/shared/programs/x86_64/anaconda2/bin:$PATH"
python -m cProfile -o gridsearch_CV.cprof gridsearch_profile.py
