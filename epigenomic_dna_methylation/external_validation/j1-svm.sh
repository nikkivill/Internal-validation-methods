#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 1
#$ -l h_vmem=5G
#$ -cwd
#$ -j y
#$ -m beas
 
# load module
module load python
 
# activate virtual environment
source /data/home/bt19531/myenv/bin/activate
 
# run python script
python 1-svm.py
