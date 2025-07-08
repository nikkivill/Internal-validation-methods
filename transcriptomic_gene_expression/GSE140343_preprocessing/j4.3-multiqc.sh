########### 4.3 - CHECK ALIGNMENT QUALITY 

#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 1
#$ -l h_vmem=5G
#$ -cwd
#$ -j y
#$ -m beas

module load python
source ~/.multiqc_env/bin/activate

module load multiqc

# directory of aligned sample files - we will use {sample}_Log.final.out files to check alignment quality 
logdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/alignment/genome_alignments
# directory for output MultiQC report
multiqcdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/fastqc/after_align 

# make sure directory exists
mkdir -p "$multiqcdir"

# run multiqc on all _Log.final.out files to summarise alignment quality results
multiqc "$logdir" --outdir "$multiqcdir"
