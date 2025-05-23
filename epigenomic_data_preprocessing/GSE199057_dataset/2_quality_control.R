######## QUALITY CONTROL 

# load necessary libraries
library(minfi)
library(RColorBrewer)

# load saved data
rgSet <- readRDS("intermediate/rgSet.rds")
targets <- readRDS("intermediate/targets.rds")

# calculate detection p-values
# detectionP() will process the full rgSet, however p-values are only returned for valid intensity data
# detP may have fewer rows than rgSet
detP <- detectionP(rgSet)
head(detP)

# examine mean detection p-values across all samples to identify any failed sample
# NOTE: p-values are so small, the threshold line is not visible here!
pal <- brewer.pal(8, "Dark2")

# save plots to pdf
pdf("../results/mean_detection_pvalues.pdf")

par(mfrow=c(1,2))

barplot(colMeans(detP), col=pal[factor(targets$Sample_Type)], las=2,
        cex.names=0.8, ylab="Mean detection p-values")
abline(h=0.01, col="red")
legend("topleft", legend=levels(factor(targets$Sample_Type)), fill=pal,
       bg="white", cex=0.6)

# zoomed in view with no p-value threshold line
barplot(colMeans(detP), col=pal[factor(targets$Sample_Type)], las=2,
        cex.names=0.8, ylim=c(0, 0.002), ylab="Mean detection p-values")
legend("topleft", legend=levels(factor(targets$Sample_Type)), fill=pal,
       bg="white", cex=0.6)

# save and close pdf of graph
dev.off()

# create quality control (QC) report PDF
qcReport(rgSet, sampNames = targets$Sample_Name, sampGroups = targets$Sample_Type,
         pdf= "../results/qcReport.pdf")

# remove poor quality samples from rgSet -  let's use a < 0.01 threshold *all samples passed
# this filters out samples (columns) which are low quality (mean detection p-value is ≥ 0.01)
keep <- colMeans(detP) < 0.01
rgSet <- rgSet[, keep]
rgSet

# remove poor quality samples from targets *all samples passed
targets <- targets[keep,]

# remove poor quality samples from detection p-value (detP) table *all samples passed
# remember detP is the subset of probes with valid detection p-values
detP <- detP[,keep]
dim(detP)

# save QC results
saveRDS(rgSet, "intermediate/rgSet_qc.rds")
saveRDS(targets, "intermediate/targets_qc.rds")
saveRDS(detP, "intermediate/detP.rds")
