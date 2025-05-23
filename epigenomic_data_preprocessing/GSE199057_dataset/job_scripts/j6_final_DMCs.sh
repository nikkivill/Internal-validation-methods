#!/bin/bash
#$ -l h_rt=01:00:00
#$ -pe smp 1
#$ -l h_vmem=8G
#$ -cwd
#$ -j y
#$ -m beas

module load R

Rscript 6_final_DMCs.R

exit
