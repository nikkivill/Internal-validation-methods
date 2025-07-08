########### 4 - FILTERING

# load necessary libraries
library(minfi)
library(RColorBrewer)
library(limma)
library(maxprobes)

# load saved data 
detP <- readRDS("intermediate/detP.rds")
mSetFun <- readRDS("intermediate/mSetFun.rds")
ann450k <- readRDS("intermediate/ann450k.rds")
targets <- readRDS("intermediate/targets_qc.rds")

# DATA EXPLORATION (before filtering)
pal <- brewer.pal(8, "Dark2")
# multi-dimensional scaling (MDS) plots are based on PCA
# MDS plots to look at largest sources of variation
# use top 10000 features

#save plots to pdf
pdf("../results/MDS_before.pdf")

par(mfrow = c(1, 1))
plotMDS(getM(mSetFun), top=10000, gene.selection="common",
        col=pal[factor(targets$Sample_Type)])
legend("topleft", legend=levels(factor(targets$Sample_Type)), text.col=pal,
       bg="white", cex=0.7)

# save and close pdf of graph
dev.off() 

# ensure probes are in the same order in mSetFun and detP objects
# we will use detP to filter probes in mSetFun
detP <- detP[match(featureNames(mSetFun), rownames(detP)),]

# remove any probes that have failed in one or more samples
# for each probe, check how many samples have a detection p-value <0.01
keep <- rowSums(detP < 0.01) == ncol(mSetFun) # keep only probes that passed in ALL samples (also removes missing values)
table(keep)

# filter the GenomicRatioSet to retain only probes that passed in ALL samples
# store in mSetFunFlt -> clean high-quality methylation dataset
mSetFunFlt <- mSetFun[keep,]
mSetFunFlt

# remove probes on the sex chromosomes as our data has males and females (confounding)
keep <- !(featureNames(mSetFunFlt) %in% ann450k$Name[ann450k$chr %in%
                                                       c("chrX", "chrY")])
table(keep)
mSetFunFlt <- mSetFunFlt[keep, ]

# remove probes where the CpG are affected by SNPs (genetic changes rather than epigenetic) - default (maf=0)
mSetFunFlt <- dropLociWithSnps(mSetFunFlt)
mSetFunFlt

# remove probes that are cross-reactive (probes demonstrated to map to multiple places in the genome)
xloci <- xreactive_probes(array_type = "450K")
length(xloci) 

mSetFunFlt <- mSetFunFlt[!(featureNames(mSetFunFlt) %in% xloci), ]

# save the filtered data
saveRDS(mSetFunFlt, "intermediate/mSetFunFlt.rds")

## DATA EXPLORATION (after filtering)
# save plots to pdf 
pdf("../results/MDS_after.pdf")

par(mfrow = c(1, 1))
plotMDS(getM(mSetFunFlt), top=10000, gene.selection="common",
        col=pal[factor(targets$Sample_Type)])
legend("topleft", legend=levels(factor(targets$Sample_Type)), text.col=pal,
       bg="white", cex=0.7)

# save and close pdf of graph
dev.off() 

