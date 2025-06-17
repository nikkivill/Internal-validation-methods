######### 1- LOAD DATA 

# set up the path to directory with raw methylation files (IDAT) - unzip files if they are zipped
dataDirectory <- "../input/GSE131013_RAW"
# check the files are there
list.files(dataDirectory, recursive=TRUE)

# load necessary libraries
library(minfi)
library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(IlluminaHumanMethylation450kmanifest)

# get EPIC annotation data
ann450k = getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
# check the data
head(ann450k)

# read in the sample sheet
targets <- read.metharray.sheet(dataDirectory, pattern="methylation1_sample_sheet.csv")
# IDAT files do not have sentrix ID or position ... so find files by the basename 
targets$Basename <- file.path(dataDirectory, targets$Basename)

# read in the raw data from the IDAT files
rgSet <- read.metharray.exp(targets=targets, force=TRUE)
rgSet

# give the samples descriptive names e.g. sample type and sample name
targets$ID <- paste(targets$Sample_Type, targets$Sample_Name, sep = ".")
sampleNames(rgSet) <- targets$ID
rgSet

# save outputs
saveRDS(rgSet, file = "intermediate/rgSet.rds")
saveRDS(targets, file = "intermediate/targets.rds")
saveRDS(ann450k, file = "intermediate/ann450k.rds")
