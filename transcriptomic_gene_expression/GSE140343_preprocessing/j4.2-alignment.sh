########### 4.2 - ALIGNMENT (STAR)

#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 1
#$ -l h_vmem=35G
#$ -cwd
#$ -j y
#$ -o j4.2-alignment_logs/$TASK_ID.log
#$ -m beas
#$ -t 1-100
#$ -tc 50

module load star

# directory of genome index
indexdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/alignment/genome_ref/star_index
# directory of trimmed fastq files
trimdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/trimmomatic
# output directory for alignment 
aligndir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/alignment/genome_alignments

# make sure directory exists
mkdir -p "$aligndir"

# get the sample ID based on the task ID
sample=$(sed -n "${SGE_TASK_ID}p" sample2_ids.txt)

# add echo to see progress
echo "Aligning $sample..."

# genome alignment, unzipping each file and outputting a BAM file per sample
STAR \
 --runMode alignReads \
 --runThreadN ${NSLOTS} \
 --genomeDir "$indexdir" \
 --readFilesIn "$trimdir"/${sample}_trimmed_1P.fastq.gz \
               "$trimdir"/${sample}_trimmed_2P.fastq.gz \
 --readFilesCommand zcat \
 --outFileNamePrefix "$aligndir"/"${sample}"_ \
 --outSAMtype BAM SortedByCoordinate

# rename the BAM file to the sample ID
mv "$aligndir"/${sample}_Aligned.sortedByCoord.out.bam \
   "$aligndir"/${sample}.bam

# clean up STAR logs
rm -f "$aligndir"/${sample}_{Log.out,Log.progress.out,SJ.out.tab}
