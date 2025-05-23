#!/bin/bash
#$ -l h_rt=01:00:00
#$ -pe smp 1
#$ -l h_vmem=18G
#$ -cwd
#$ -j y
#$ -m beas

module load R

Rscript 2_quality_control.R 

exit
