#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 4
#$ -l h_vmem=8G
#$ -cwd
#$ -j y
#$ -m beas
 
# load module
module load python
 
# activate virtual environment
source /data/home/bt19531/myenv/bin/activate
 
# run python script
python 3-rf.py
