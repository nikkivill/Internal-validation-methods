#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 1
#$ -l h_vmem=20G
#$ -cwd
#$ -j y
#$ -m beas

module load R

Rscript 7.2-deseq.R
exit
