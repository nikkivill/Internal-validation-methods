# identify top 20 DMCs 

# load saved data
targets <- readRDS("targets_qc.rds")
bVals <- readRDS("bVals.rds")
DMCs <- readRDS("DMCs.rds")
ALL <- readRDS("ALL.rds")

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
cat("Rows in DMCsfilt (FDR < 0.05): ", nrow(DMCsfilt), "\n")

# rank by absolute delta beta (descending)
DMCsfilt_ordered <- DMCsfilt[order(-abs(DMCsfilt$delta_beta_vals)), ]

# take top 20 DMCs
top20_DMCs <- head(DMCsfilt_ordered, 20)

# print results
cat("Top 20 DMCs ranked by absolute delta beta (FDR < 0.05):\n")
print(top20_DMCs[, c("delta_beta_vals", "adj.P.Val")])

# save filtered beta values for top 20 CpGs
finalbVals_top20 <- bVals[rownames(top20_DMCs), ]

########### VISUALISE USING VOLCANO PLOT
  
# load necessary libraries
library(ggplot2)
library(ggrepel)

ALL$delta_beta_vals <- delta_beta[rownames(ALL)]
ALL$neg_log10_FDR <- -log10(ALL$adj.P.Val)

# CpGs with FDR >0.05
ALL$Direction <- "Not significant"
# CpGs with FDR <0.05
ALL$Direction[ALL$adj.P.Val < 0.05] <- "Significant"
# top 20 CpGs and their methylation direction 
ALL$Direction[rownames(ALL) %in% rownames(finalbVals_top20) & ALL$delta_beta_vals > 0] <- "Hypermethylated"
ALL$Direction[rownames(ALL) %in% rownames(finalbVals_top20) & ALL$delta_beta_vals < 0] <- "Hypomethylated"

# convert to factor to control legend order
ALL$Direction <- factor(ALL$Direction, levels = c("Hypermethylated",
                                                  "Hypomethylated",
                                                  "Significant",
                                                  "Not significant"))

# add labels for top 20 CpGs 
ALL$label <- ifelse(ALL$Direction %in% c("Hypermethylated", "Hypomethylated"),
                    rownames(ALL), NA)

# plot volcano plot
ggplot(ALL,
       aes(x=delta_beta_vals,
           y=neg_log10_FDR,
           color=Direction)) +
  geom_point(alpha=0.6,
             size=1) +
  geom_text_repel(aes(label = label),
                  size=5,
                  nudge_y=0,
                  max.overlaps=Inf,
                  box.padding=0.6,
                  point.padding=0.1,
                  force=3,
                  force_pull=0.05,
                  min.segment.length = 0.2,
                  segment.angle = 30,
                  show.legend=FALSE)+
  scale_color_manual(values = c("Hypermethylated"="red",
                                "Hypomethylated"="blue",
                                "Not significant"="azure2",
                                "Significant"="azure3")) +
  geom_hline(yintercept=-log10(0.05),
             linetype="dashed",
             color="black") +
  scale_y_continuous(breaks = seq(0, ceiling(max(ALL$neg_log10_FDR, na.rm = TRUE)), by = 10)) +
  labs(x="Δβ (Tumour vs Normal)",
       y="-log10 (FDR)",
       color = NULL) +
  guides(color = guide_legend(override.aes = list(size = 4))) +
  theme_minimal() +
  theme(legend.position='top',
        legend.text = element_text(size = 16),
        axis.text = element_text(size = 16),       
        axis.title = element_text(size = 16))      
          
 
