########### 3.3 - QUALITY CONTROL PT2 (AFTER TRIMMING)

#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 1
#$ -l h_vmem=2G
#$ -cwd
#$ -j y
#$ -m beas

module load python
source ~/.multiqc_env/bin/activate

module load multiqc

# input directory of fastqc files and output directory for MultiQC report
fastqcdir=/data/scratch/bt19531/23-06-25_transcriptomic/data_preprocessing/tmp/fastqc/after_trim

# run multiqc on all fastqc files to summarise all fastqc reports
multiqc \
 "$fastqcdir" \
 --outdir "$fastqcdir"
