########### 3.1 - ADAPTOR AND QUALITY TRIMMING AND REMOVAL OF SHORT READS (TRIMMOMATIC)

#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 1
#$ -l h_vmem=12G
#$ -cwd
#$ -j y
#$ -o j3.1-trimmomatic_logs/$TASK_ID.log
#$ -m beas
#$ -t 1-246
#$ -tc 50

module load trimmomatic 

# directory of fastq files
fastqdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/input/fastq
# directory of adapter file 
adaptdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/trimmomatic/adapters
# output directory for the trimmomatic files
trimdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/trimmomatic

# make sure directories exist
mkdir -p "$adaptdir" "$trimdir"

# download the adapter file once (stops from downloading it every run)
[ -s "$adaptdir"/TruSeq3-PE.fa ] || \
  wget -q -O "$adaptdir"/TruSeq3-PE.fa \
  https://raw.githubusercontent.com/timflutre/trimmomatic/master/adapters/TruSeq3-PE.fa

# get the sample ID based on the task ID
sample=$(sed -n "${SGE_TASK_ID}p" sample2_ids.txt)

# fastq files are paired
fq1="$fastqdir/${sample}_pass_1.fastq.gz"
fq2="$fastqdir/${sample}_pass_2.fastq.gz"

# adaptor and quality trimming and removal of short end reads using trimmomatic in paired-end mode
trimmomatic PE \
 -threads ${NSLOTS} \
 "$fq1" "$fq2" \
 "$trimdir"/${sample}_trimmed_1P.fastq.gz \
 /dev/null \
 "$trimdir"/${sample}_trimmed_2P.fastq.gz \
 /dev/null \
 ILLUMINACLIP:"$adaptdir/TruSeq3-PE.fa":2:30:10:2:True \
 LEADING:3 TRAILING:3 MINLEN:36
