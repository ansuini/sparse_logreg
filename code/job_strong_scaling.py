#!/bin/bash
#PBS -N gridsearch
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=20
#PBS -T flush_cache
#PBS -q regular

NAME=gridsearch
WD='/scratch/ansuini/repositories/machine_learning/sparse_logreg/code'
cd $WD
echo 'Working directory is' $WD
rm -f gridsearch*

python parallel_elasticnet_gridsearch_CV.py >> out
