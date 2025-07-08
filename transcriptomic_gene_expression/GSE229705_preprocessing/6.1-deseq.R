########### 1 - DIFFERENTIAL EXPRESSION ANALYSIS PT1 (DESEQ)

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

# set sample type as a factor and make normal the reference
metadata$Sample_Type <- factor(metadata$Sample_Type, levels = c("Normal", "Tumor")) 
levels(metadata$Sample_Type)

# check sample order is the name in metadata and counts 
all(rownames(metadata) == colnames(counts))  # Should be TRUE

# rename column names in counts to include sample type
colnames(counts) <- paste(metadata$Sample_Type, rownames(metadata), sep = ".")

# update metadata rownames too 
rownames(metadata) <- colnames(counts)

# construct DESeqDataSet object from read counts and metadata
dds <- DESeqDataSetFromMatrix(countData = counts,
                                  colData = metadata,
                                  design = ~ Sample_Source + Sample_Type)
dds

# filter out rows that have 0 counts or single counts - not informative 
nrow(dds)
dds <- dds[rowSums(counts(dds)) > 1, ]
nrow(dds)

# run DESeq on read counts 
dds <- DESeq(dds)

# save objects
saveRDS(dds, "dds.rds")

