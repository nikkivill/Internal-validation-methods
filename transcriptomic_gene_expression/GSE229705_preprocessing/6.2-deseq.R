########### 6.2 - DIFFERENTIAL EXPRESSION ANALYSIS PT2 (DESEQ)

# import necessary libraries
library(DESeq2)

# load objects
dds <- readRDS("dds.rds")

# check results
res <- results(dds, alpha=0.05) 
summary(res)

# how many genes passed the 0.05 FDR threshold
table(res$padj < 0.05)
# filter results for FDR < 0.05
res_filt <- res[!is.na(res$padj) & res$padj < 0.05, ]

# rank results by log2 to find DEGs
res_ranked <- res_filt[order(abs(res_filt$log2FoldChange), decreasing = TRUE), ]

# select top 20 DEGs
top20 <- head(res_ranked, 20)
# extract top 20 DEG names
top20_genes <- rownames(top20)

# normalise the data using vst
vst_dds <- vst(dds, blind = FALSE)
vst_mat <- assay(vst_dds)  # DEGs x samples

# subset for top 20 DEGs
top20_DEGs <- vst_mat[top20_genes, ]

# save top 20 DEGs
writeLines(top20_genes, "top20_DEGs.txt")
# save top 20 DEGs matrix
write.csv(as.data.frame(top20_DEGs), "top20_DEGs.csv")
