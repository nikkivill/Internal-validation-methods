
########### VISUALISE TOP 20 DEGs USING VOLCANO PLOT

library(ggplot2)
library(ggrepel)

# load data 
lfc_shrunk <- read.csv("DEG_results_shrunken_log2FC.csv", row.names = 1)
top20_DEGs <- read.csv("top20_DEGs.csv", row.names = 1)
genes <- read.csv("gene_names.csv", row.names = 1)

# remove version from gene ids
clean_ids <- function(ids) {
  sub("\\..*$", "", ids)
}

# apply to lfc_shrunk and top20_DEGs
rownames(lfc_shrunk) <- clean_ids(rownames(lfc_shrunk))
rownames(top20_DEGs) <- clean_ids(rownames(top20_DEGs))

head(rownames(lfc_shrunk))
head(rownames(top20_DEGs))

# df for plotting 
df <- lfc_shrunk
df$Ensembl   <- rownames(df)
df$negLog10FDR <- -log10(df$padj)         
df$top20     <- df$Ensembl %in% rownames(top20_DEGs)

# get gene name
df$symbol <- genes[df$Ensembl, "Gene.name"]
sum(!is.na(df$symbol))

# define regulation of gene 
df$regulation <- "Not significant"
# genes with FDR <0.05
df$regulation[df$padj < 0.05] <- "Significant"
# top 20 DEGs and their regulation direction
df$regulation[rownames(df) %in% rownames(top20_DEGs) & df$log2FoldChange > 0] <- "Upregulated"
df$regulation[rownames(df) %in% rownames(top20_DEGs) & df$log2FoldChange < 0] <- "Downregulated"

# convert to factor to control legend order
df$regulation <- factor(df$regulation, levels = c("Upregulated",
                                                  "Downregulated",
                                                  "Significant",
                                                  "Not significant"))
# for centering graph 
xmax <- max(abs(df$log2FoldChange), na.rm = TRUE)

ggplot(df,
       aes(x=log2FoldChange,
           y=negLog10FDR,
           color=regulation)) +
  geom_point(size=1.5,
             alpha=0.6) +
  scale_color_manual(values = c("Upregulated"="red",
                                "Downregulated"="blue",
                                "Significant"="azure3",
                                "Not significant"="azure2")) +
  geom_hline(yintercept = -log10(0.05),
             linetype="dashed",
             color="black") +
  geom_text_repel(
    data = subset(df, regulation %in% c("Upregulated","Downregulated")),
    aes(label = symbol),
    size=5,
    box.padding=0.3,
    point.padding=0.2,
    max.overlaps=Inf,
    segment.color=NA,
    show.legend=FALSE) +
  scale_x_continuous(limits = c(-xmax, xmax)) +
  labs(x="Log2 Fold Change (Tumour vs Normal)",
       y="-log10 (FDR)") +
  scale_y_continuous(breaks = seq(0, ceiling(max(df$negLog10FDR, na.rm = TRUE)), by = 20)) +
  labs(color = NULL) +
  guides(color = guide_legend(override.aes = list(size = 4))) +
  theme_minimal() +
  theme(legend.position = 'top',
        legend.text = element_text(size = 16),
        axis.text = element_text(size = 16),       
        axis.title = element_text(size = 16)) 










