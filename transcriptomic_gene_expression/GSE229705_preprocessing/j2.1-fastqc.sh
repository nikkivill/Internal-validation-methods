########### 2.1 -  QUALITY CONTROL PT1 (FASTQC)

#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 1
#$ -l h_vmem=5G
#$ -cwd
#$ -j y
#$ -o fastqc_logs/$TASK_ID.log
#$ -m beas
#$ -t 1-246
#$ -tc 50

module load fastqc

# directory of the fastq files
fastqdir=/data/scratch/bt19531/23-06-25_transcriptomic/data_preprocessing/input/fastq
# output directory for fastqc zip and html files 
fastqcdir=/data/scratch/bt19531/23-06-25_transcriptomic/data_preprocessing/tmp/fastqc/before_trim

# make sure directory exists
mkdir -p "$fastqcdir"

# get the sample ID based on the task ID
sample=$(sed -n "${SGE_TASK_ID}p" sample_ids.txt)

# quality control using fastqc on paired reads
for read in 1 2; do

    fastq="$fastqdir/${sample}_pass_${read}.fastq.gz"
    fastqc --nogroup --outdir "$fastqcdir" "$fastq"

done
