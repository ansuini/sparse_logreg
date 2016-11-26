#!/bin/bash
#PBS -N parallel_gridsearch
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=20
#PBS -T flush_cache
#PBS -q regular

NAME=parallel_gridsearch
WD='/scratch/ansuini/repositories/machine_learning/sparse_logreg/code'
cd $WD
echo 'Working directory is' $WD
rm -f $WD/parallel_gridsearch*

python logreg_elasticnet_gridsearch_CV.py >> out
