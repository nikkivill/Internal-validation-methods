########### 6.1 - DIFFERENTIAL EXPRESSION ANALYSIS (DESEQ)

# import necessary libraries
library(DESeq2)

# path to files
counts_path <- "/data/scratch/bt19531/23-06-25_transcriptomic/differential_analysis/external/input/external_counts.txt"
metadata_path <- "/data/scratch/bt19531/23-06-25_transcriptomic/differential_analysis/external/input/external_metadata.csv"

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

# normalise the data using vst
vst_dds <- vst(dds, blind = FALSE)
vst_mat <- assay(vst_dds)  # DEGs x samples

# read top 20 DEG names from file
top20_genes <- readLines("/data/scratch/bt19531/23-06-25_transcriptomic/differential_analysis/external/input/top20_DEGs.txt")

# check how many of the top 20 genes are in external dataset
top20_genes[!top20_genes %in% rownames(vst_mat)]
# keep only the genes that exist in vst_mat
present_genes <- top20_genes[top20_genes %in% rownames(vst_mat)]
top20_DEGs <- vst_mat[present_genes, ]

# save top 20 DEGs matrix
write.csv(as.data.frame(top20_DEGs), "external_top20_DEGs.csv")
