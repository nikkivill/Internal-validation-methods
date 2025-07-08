########### 3.2 - QUALITY CONTROL PT1 (AFTER TRIMMING)

#!/bin/bash
#$ -l h_rt=240:00:00 
#$ -pe smp 1
#$ -l h_vmem=5G
#$ -cwd
#$ -j y
#$ -o j3.2-fastqc_logs/$TASK_ID.log
#$ -m beas
#$ -t 1-100
#$ -tc 50

module load fastqc

# directory of trimmed fastq files
trimdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/trimmomatic
# output directory for fastqc zip and html files (after trimming)
fastqcdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/fastqc/after_trim

# make sure directory exist
mkdir -p "$fastqcdir"

# get the sample ID based on the task ID
sample=$(sed -n "${SGE_TASK_ID}p" sample2_ids.txt)

# quality control using fastqc on trimmed paired reads
for read in 1 2; do

    fastq="$trimdir/${sample}_trimmed_${read}P.fastq.gz"
    fastqc \
    --nogroup \
    --outdir "$fastqcdir" \
    "$fastq"

done
