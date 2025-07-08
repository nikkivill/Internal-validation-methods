########### 5 - READ COUNT (FEATURECOUNTS)

#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 8
#$ -l h_vmem=10G
#$ -cwd
#$ -j y
#$ -m beas

module load subread

# directory of genome annotation (gtf)
anndir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/alignment/genome_ref
# directory of aligned BAM files
bamdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/alignment/genome_alignments
# directory for output read count
outdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/read_counts
# directory for output temp files
tempdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/read_counts/tempdir

# make sure directory exist
mkdir -p "$outdir" "$tempdir"

cd "$bamdir"

featureCounts \
 -T ${NSLOTS} \
 -p --countReadPairs \
 -s 0 \
 -t exon \
 -g gene_id \
 -a "$anndir"/gencode.v45.annotation.gtf \
 --tmpDir "$tempdir" \
 -o "$outdir"/counts.txt \
 *.bam

# remove all temporary files 
rm -r "$tempdir"
