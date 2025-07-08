########### 4.1 - GENOME REFERENCE (STAR)

#!/bin/bash
#$ -l h_rt=240:00:00
#$ -pe smp 1
#$ -l h_vmem=40G
#$ -cwd
#$ -j y
#$ -m beas

module load star

# directory of genome reference
refdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/alignment/genome_ref
# directory of index files 
indexdir=/data/scratch/bt19531/23-06-25_transcriptomic/external_preprocessing/tmp/alignment/genome_ref/star_index

# make sure directories exist
mkdir -p "$refdir" "$indexdir"

# download genome assembly reference
wget -q -O "$refdir"/GRCh38.primary_assembly.genome.fa.gz https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz 
# download annotation gtf file 
wget -q -O "$refdir"/gencode.v45.annotation.gtf.gz https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.annotation.gtf.gz

gunzip "$refdir"/GRCh38.primary_assembly.genome.fa.gz
gunzip "$refdir"/gencode.v45.annotation.gtf.gz

# genome indexing
STAR \
 --runMode genomeGenerate \
 --runThreadN ${NSLOTS} \
 --genomeDir "$indexdir" \
 --genomeFastaFiles "$refdir"/GRCh38.primary_assembly.genome.fa \
 --sjdbGTFfile "$refdir"/gencode.v45.annotation.gtf 
