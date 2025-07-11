########### 6 - FINAL DMCs

########### FILTERING FOR TOP 20 CpGs 

# load saved data
targets <- readRDS("intermediate/targets_qc.rds")
bVals <- readRDS("intermediate/bVals.rds")
DMCs <- readRDS("intermediate/DMCs.rds")
ALL <- readRDS("intermediate/ALL.rds")

# per CpG (row) in my beta values df - calculate the mean difference between tumor and normal samples
tumor_samples <- targets$ID[targets$Sample_Type == "Tumor"]
normal_samples <- targets$ID[targets$Sample_Type == "Normal"]

tumor_means <- rowMeans(bVals[, tumor_samples])
normal_means <- rowMeans(bVals[, normal_samples])

delta_beta <- tumor_means - normal_means

# add delta_beta values as column
names(delta_beta) <- rownames(bVals)
DMCs$delta_beta_vals <- delta_beta[rownames(DMCs)]

# filter DMCs - only FDR < 0.05 
DMCsfilt <- DMCs[DMCs$adj.P.Val < 0.05, ]
cat("▶Rows in DMCsfilt (FDR < 0.05): ", nrow(DMCsfilt), "\n")

# rank by absolute delta beta (descending)
DMCsfilt_ordered <- DMCsfilt[order(-abs(DMCsfilt$delta_beta_vals)), ]

# take top 20 DMCs
top20_DMCs <- head(DMCsfilt_ordered, 20)

# print results
cat("Top 20 DMCs ranked by absolute delta beta (FDR < 0.05):\n")
print(top20_DMCs[, c("delta_beta_vals", "adj.P.Val")])

# save filtered beta values for top 20 CpGs
finalbVals_top20 <- bVals[rownames(top20_DMCs), ]
saveRDS(finalbVals_top20, "../results/top20_bVals.rds")
