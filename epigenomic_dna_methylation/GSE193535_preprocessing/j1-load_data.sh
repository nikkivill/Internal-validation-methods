#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 1
#$ -l h_vmem=10G
#$ -cwd
#$ -j y
#$ -m beas

module load R

Rscript 1-load_data.R
exit
