#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 6
#$ -l h_vmem=4G
#$ -cwd
#$ -j y
#$ -m beas
 
# load module
module load python
 
# activate virtual environment
source /data/home/bt19531/myenv/bin/activate
 
# run python script
python 4-mlp.py
