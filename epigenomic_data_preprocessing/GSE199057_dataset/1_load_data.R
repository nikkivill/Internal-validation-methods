########## LOAD DATA 

# set up the path to directory with raw methylation files (IDAT)
dataDirectory <- "../input"
# check the files are there
list.files(dataDirectory, recursive = TRUE)

# load necessary libraries
library(minfi)
library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
library(IlluminaHumanMethylationEPICmanifest)

# get EPIC annotation data
annEPIC = getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
# check the data
head(annEPIC)

# read in the sample sheet
targets <- read.metharray.sheet(dataDirectory, pattern="methylation_sample_sheet.csv")

# read in the raw data from the IDAT files
rgSet <- read.metharray.exp(targets=targets)
rgSet

# give the samples descriptive names e.g. sample type and sample name
targets$ID <- paste(targets$Sample_Type, targets$Sample_Name, sep = ".")
sampleNames(rgSet) <- targets$ID
rgSet

# save outputs
saveRDS(rgSet, file = "intermediate/rgSet.rds")
saveRDS(targets, file = "intermediate/targets.rds")
saveRDS(annEPIC, file = "intermediate/annEPIC.rds")
