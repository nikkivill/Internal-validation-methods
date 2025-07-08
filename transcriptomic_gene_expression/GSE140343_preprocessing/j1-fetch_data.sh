########### 1 -  FETCH DATA (SRA TOOLKIT)

#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 1
#$ -l h_vmem=6G
#$ -cwd
#$ -j y
#$ -o /dev/null
#$ -e /dev/null
#$ -m beas
#$ -t 1-100
#$ -tc 50

module load sra-tools

# output directory for SRA files
sradir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/input/sra
# output directory for fastq files
fastqdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/input/fastq

# make sure directories exist 
mkdir -p "$sradir" "$fastqdir"

# get the sample ID based on the task ID
sample=$(sed -n "${SGE_TASK_ID}p" sample2_ids.txt)

# download SRA files using the SRA run identifiers
prefetch "$sample" -O "$sradir" 

# convert SRA data into FASTQ format
fastq-dump \
 --outdir "$fastqdir" \
 --gzip \
 --skip-technical \
 --readids \
 --read-filter pass \
 --dumpbase \
 --split-3 \
 --clip \
 "$sradir"/"$sample"/"$sample".sra

# delete SRA folder for this sample after
rm -rf "$sradir/$sample"
