########### 5 - DMCs

########### CALCULATE M-VALUES AND BETA VALUES 

# load necessary libraries
library(minfi)
library(limma)
library(RColorBrewer)

# load saved data
mSetFunFlt <- readRDS("intermediate/mSetFunFlt.rds")
targets <- readRDS("intermediate/targets_qc.rds")
# rename the first column of targets (R has changed it)
colnames(targets)[1] <- "Sample_Source" 

# calculate M-values for statistical analysis
mVals <- getM(mSetFunFlt)
head(mVals[,1:5])

# calculate beta values 
bVals <- getBeta(mSetFunFlt)
head(bVals[,1:5])

# density plots for Beta and M values
pdf("../results/b_m_vals.pdf")

par(mfrow=c(1,2))
densityPlot(bVals, sampGroups=targets$Sample_Type, main="Beta values",
            legend=FALSE, xlab="Beta values")
legend("top", legend = levels(factor(targets$Sample_Type)),
       text.col=brewer.pal(8,"Dark2"))
densityPlot(mVals, sampGroups=targets$Sample_Type, main="M-values",
            legend=FALSE, xlab="M values")
legend("topleft", legend = levels(factor(targets$Sample_Type)),
       text.col=brewer.pal(8,"Dark2"))

dev.off()

# save M-values and B-values
saveRDS(mVals, "intermediate/mVals.rds")
saveRDS(bVals, "intermediate/bVals.rds") 

######### PROBE-WISE (CpG) DIFFERENTIAL METHYLATION ANALYSIS - FINDING SIGNIFICANT CpGs

# because our samples are paired (tumor and normal pairs come from one individual)
# we need to account for individual to individual variation 
# we can use limma and the M-values to find significant CpGs then filter our beta values for these

# factor of interest is tumor vs normal(2 factor levels)
sampleType <- factor(targets$Sample_Type)
# the effect we need to account for is the individual where samples came from (no. individuals = factor levels)
individual <- factor(targets$Sample_Source)

# ensure numbers match for all 
cat("nrow(targets):    ", nrow(targets), "\n",
    "length(sampleType):", length(sampleType), "\n",
    "length(individual):", length(individual), "\n")
# create a design matrix - use '0' to drop intercept so that each sample type and individual has a column
design <- model.matrix(~0+sampleType+individual, data=targets)

# make.names is used as individual IDs are numeric 
# individual 1 is dropped to serve as a baseline to avoid perfect multicollinearity
# *when one variable is the perfect linear of another variable - model cannot estimate coefficients 
colnames(design) <- make.names(c(levels(sampleType), levels(individual)[-1]))

# fit the linear model
fit <- lmFit(mVals, design)
# contrast of interest (tumor-normal) -> tests with-in pair differences
contMatrix <- makeContrasts(Tumor-Normal, levels=design)
# apply the contrast to the fitted model to recalculate coefficients for specified contrast
fit2 <- contrasts.fit(fit, contMatrix)
# obtain more stable variance estimates (good for small sample sizes)
fit2 <- eBayes(fit2)
# get summary for DMCs for each direction (e.g. down for hypo in tumor, up for hyper in tumor)
# by default: method is separate, FDR <0.05, adjust method is Benjamini-Hochberg 
summary(decideTests(fit2))
# extract table of DMCs for the first (and only) contrast, return results for all probes, filter probes with FDR-adjusted p-value
DMCs <- topTable(fit2, coef=1, number=Inf, p.value=0.05)

# for volcano plot later 
ALL <- topTable(fit2, coef=1, number=Inf)

# print number of CpGs in ALL
cat("Number of CpGs:", nrow(ALL), "\n")
# print number of CpGs in DMCs
cat("Number of significant CpGs:", nrow(DMCs), "\n")

# save the data
saveRDS(DMCs, "intermediate/DMCs.rds")
saveRDS(ALL, "intermediate/ALL.rds")

# plot the top 4 most significantly differentiated CpGs
pdf("../results/top_4_DMCs.pdf")

par(mfrow = c(2, 2))
sapply(rownames(DMCs)[1:4], function(cpg) {plotCpg(bVals, cpg = cpg, pheno = targets$Sample_Type,
                                                   ylab = "Beta values")})

dev.off()                                        
