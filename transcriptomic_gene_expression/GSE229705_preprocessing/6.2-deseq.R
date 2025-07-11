########### 6.2 - DIFFERENTIAL EXPRESSION ANALYSIS PT 2 (DESEQ)

# import necessary libraries
library(DESeq2)
library(apeglm)

# load objects
dds <- readRDS("dds.rds")

# check results using FDR 0.05
res <- results(dds, alpha=0.05)
res
# how many genes passed the 0.05 FDR threshold
sum(res$padj < 0.05, na.rm=TRUE)

# get coefficient name
resultsNames(dds)
# shrinkage of log2fc for ranking genes using apeglm
resLFC <- lfcShrink(dds, coef="Sample_Type_Tumor_vs_Normal", type="apeglm")
resLFC

# filter by FDR 0.05 and remove NA values
resLFC_filt <- resLFC[!is.na(resLFC$padj) & resLFC$padj < 0.05, ]

# rank by absolute log2fc
resLFC_ranked <- resLFC_filt[order(abs(resLFC_filt$log2FoldChange), decreasing = TRUE), ]
# extract top 20 ranked genes by log2fc
top20 <- head(resLFC_ranked, 20)
top20
# extract top 20 DEG names
top20_genes <- rownames(top20)

# get VST matrix for top genes (needed for machine learning input)
vst_dds <- vst(dds, blind = FALSE)
vst_mat <- assay(vst_dds)  # DEGs x samples

# subset for top 20 DEGs
top20_DEGs <- vst_mat[top20_genes, ]

# save top 20 DEGs as text file 
writeLines(top20_genes, "top20_DEGs.txt")
# save top 20 DEGs matrix
write.csv(as.data.frame(top20_DEGs), "top20_DEGs.csv")
