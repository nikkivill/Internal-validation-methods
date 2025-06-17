########### 6 - FILTERING FOR THE INTERNAL DATASET TOP 20 DMCs

# load saved data
targets <- readRDS("intermediate/targets_qc.rds")
bVals <- readRDS("intermediate/bVals.rds")
DMCs <- readRDS("intermediate/DMCs.rds")
ALL <- readRDS("intermediate/ALL.rds")

# load the existing list of CpGs you want to keep
top20_bVals <- readRDS("../../internal_data/results/top20_bVals.rds")

# get the CpG names from top20_bvals file
cpgs_to_keep <- rownames(top20_bVals)

# filter your bVals to only include these CpGs
filtered_bVals <- bVals[cpgs_to_keep, ]

cat("Number of DMCs kept:", nrow(filtered_bVals), "\n")

# print names of retained CpGs
cat("DMCs:\n")
print(rownames(filtered_bVals))

saveRDS(filtered_bVals, "../results/external_top20_bVals.rds")
