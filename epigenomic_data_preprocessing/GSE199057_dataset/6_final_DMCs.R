######## FILTERING FOR SIGNIFICANT CpGs

# load necessary libraries
library(ggplot2)

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

# add this as a column in my DMCs using row names (CpGs)
names(delta_beta) <- rownames(bVals)
DMCs$delta_beta_vals <- delta_beta[rownames(DMCs)]

# add a threshold of 0.2 and FDR or 0.05 and filter DMCs using this -> DMCsfilt
DMCsfilt <- DMCs[abs(DMCs$delta_beta_vals) > 0.2 & DMCs$adj.P.Val < 0.05, ]

# filter original beta values using DMCsfilt -> finalbVals
finalbVals <- bVals[rownames(DMCsfilt), ]

# print number of CpGs in finalbVals
cat("Number of significant CpGs:", nrow(finalbVals), "\n")

# save data
saveRDS(finalbVals, "../results/finalbVals.rds")

### VISUALISE USING VOLCANO PLOTS 

ALL$delta_beta_vals <- delta_beta[rownames(ALL)]

ALL$neg_log10_FDR <- -log10(ALL$adj.P.Val)

ALL$Significant <- ifelse(abs(ALL$delta_beta_vals) > 0.2 & ALL$adj.P.Val < 0.05,
                          "Significant", "Not Significant")

pdf("../results/DMCs_volcano_plot.pdf")

ggplot(ALL, aes(x=delta_beta_vals, y=neg_log10_FDR, color=Significant)) +
  geom_point(alpha=0.6, size=1) +
  scale_color_manual(values = c("Significant"="blue", "Not Significant"="black")) +
  geom_vline(xintercept = c(-0.2, 0.2), linetype="dashed", color="red") +
  geom_hline(yintercept = -log10(0.05), linetype="dashed", color="red") +
  labs(title="Volcano plot of differentially methylated CpGs",
       x="Delta Beta",
       y="-log10 (FDR)") +
  theme_minimal()

dev.off()
