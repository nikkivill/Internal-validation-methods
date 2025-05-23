########## NORMALISATION

# load necessary libraries 
library(minfi)
library(RColorBrewer)
library(limma)

# load saved data 
rgSet <- readRDS("intermediate/rgSet_qc.rds")
targets <- readRDS("intermediate/targets_qc.rds")

# normalise the data using preprocessFunnorm (best for cancer/normal sample types)
# *adjusts for unwanted variation but keeps biological differences
# methylated/unmethylated intensity values have been converted...
# GenomicRatioSet object -> M-values (log2 methylated/unmethylated), beta values (0-1), genomic coordinates (per CpG)
mSetFun <- preprocessFunnorm(rgSet)

# create a MethylSet object from the raw data for plotting
mSetRaw <- preprocessRaw(rgSet)

# save the data 
saveRDS(mSetFun, "intermediate/mSetFun.rds")
saveRDS(mSetRaw, "intermediate/mSetRaw.rds")

# visualise the data beta values before normalisation
pal <- brewer.pal(8, "Dark2")

# save plots to pdf 
pdf("../results/normalisation_density_plot.pdf")

par(mfrow=c(1,2))
densityPlot(getBeta(mSetRaw), sampGroups = targets$Sample_Type, main="Raw", legend=FALSE)
legend("top", legend = levels(factor(targets$Sample_Type)),
       text.col=brewer.pal(8, "Dark2"))
# visualise the data after normalisation
densityPlot(getBeta(mSetFun), sampGroups = targets$Sample_Type,
            main="Normalised", legend=FALSE)
legend("top", legend = levels(factor(targets$Sample_Type)), 
       text.col = brewer.pal(8, "Dark2"))

# save and close pdf of graph
dev.off()
