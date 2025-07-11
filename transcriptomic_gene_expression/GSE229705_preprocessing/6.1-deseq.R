########### 6.1 - DIFFERENTIAL EXPRESSION ANALYSIS PT 1 (DESEQ)

# import necessary libraries
library(DESeq2)

# path to files
counts_path <- "/data/scratch/bt19531/23-06-25_transcriptomic/differential_analysis/internal/input/internal_counts.txt"
metadata_path <- "/data/scratch/bt19531/23-06-25_transcriptomic/differential_analysis/internal/input/internal_metadata.csv"

# import read counts - ignoring first line
raw_counts <- read.delim(counts_path, comment.char = "#", header = TRUE)
# make gene ID as the rows
rownames(raw_counts) <- raw_counts$Geneid
# remove columns that aren't samples 
counts <- raw_counts[, 7:ncol(raw_counts)]
# remove ".bam" from sample names
colnames(counts) <- sub("\\.bam$", "", colnames(counts))
head(counts)

# import metadata - make sample name as the rows
metadata <- read.csv(metadata_path, row.names = 1) 
head(metadata)

# check sample order is the name in metadata and counts 
all(rownames(metadata) == colnames(counts))  

# rename column names in counts to include sample type
colnames(counts) <- paste(metadata$Sample_Type, rownames(metadata), sep = ".")

# update metadata rownames too 
rownames(metadata) <- colnames(counts)

# setting factors levels for deseq
metadata$Sample_Source <- factor(metadata$Sample_Source)
metadata$Sample_Type <- factor(metadata$Sample_Type, levels = c("Normal", "Tumor"))

# construct DESeqDataSet object from read counts and metadata
dds <- DESeqDataSetFromMatrix(countData = counts,
                                  colData = metadata,
                                  design = ~ Sample_Source + Sample_Type)
dds

# keep rows which have minimum count of 10 for a minimium number of samples (smallest group size)
nrow(dds)
smallestgroup <- 123 # smallest group size 
keep <- rowSums(counts(dds) >= 10) >= smallestgroup
dds <- dds[keep, ]
nrow(dds)

# run DESeq on read counts 
dds <- DESeq(dds)

# save objects
saveRDS(dds, "dds.rds")
