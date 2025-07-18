######## Load the data

# import necessary libraries
import pyreadr
import pandas as pd
import joblib

# load data - CpGs are already the index
data = pyreadr.read_r("input/top20_bVals.rds") # read in rds file 
df = data[None] # methylation beta value matrix
print(df.shape)
df.head()

# transpose the dataframe
df_t = df.T
print(df_t.shape)
df_t.head()

# load the metadata
metadata = pd.read_csv("input/methylation1_sample_sheet.csv")
metadata.head()
# ensure both columns are strings
metadata['Sample_Type'] = metadata['Sample_Type'].astype(str)
metadata['Sample_Name'] = metadata['Sample_Name'].astype(str)
# create new column that matches methylation df column names (for paired splitting) 
metadata['Sample'] = metadata['Sample_Type'] + '.' + metadata['Sample_Name']
# set as index
metadata.set_index('Sample', inplace=True)
# make sure both indexes matches
metadata = metadata.loc[df_t.index]
print(metadata.shape)
metadata.head()

# group paired samples together using metadata
sample_ids = df_t.index
# look up sample_source values for sample_ids
groups = metadata.loc[sample_ids, 'Sample_Source'].values 
print(groups)

### specify features and labels
X = df_t
# create binary labels with 0 = Normal, 1 = Tumor 
# convert to pandas.Series to use .iloc later
Y = pd.Series([0 if idx.startswith("Normal") else 1 for idx in X.index], index=X.index)

# save objects for later
joblib.dump(X,"input/X.pkl")
joblib.dump(Y, "input/Y.pkl")
joblib.dump(groups,"input/groups.pkl")





######### Load the external data 

# load external data - CpGs are already the index
data = pyreadr.read_r("input/external_top20_bVals.rds") # read in rds file 
ex_df = data[None] # methylation beta value matrix
print(ex_df.shape)
ex_df.head()

# transpose the dataframe
ex_df_t = ex_df.T
print(ex_df_t.shape)
ex_df_t.head()

# load the metadata
ex_metadata = pd.read_csv("input/methylation2_sample_sheet.csv")
ex_metadata.head()
# ensure both columns are strings
ex_metadata['Sample_Type'] = ex_metadata['Sample_Type'].astype(str)
ex_metadata['Sample_Name'] = ex_metadata['Sample_Name'].astype(str)
# create new column that matches methylation df column names (for paired splitting) 
ex_metadata['Sample'] = ex_metadata['Sample_Type'] + '.' + ex_metadata['Sample_Name']
# set as index
ex_metadata.set_index('Sample', inplace=True)
# make sure both indexes matches
ex_metadata = ex_metadata.loc[ex_df_t.index]
print(ex_metadata.shape)
ex_metadata.head()

# group paired samples together using metadata
ex_sample_ids = ex_df_t.index
# look up sample_source values for sample_ids
ex_groups = ex_metadata.loc[ex_sample_ids, 'Sample_Source'].values 
print(ex_groups)

### specify features and labels
ex_X = ex_df_t
# create binary labels with 0 = Normal, 1 = Tumor 
# convert to pandas.Series to use .iloc later
ex_Y = pd.Series([0 if idx.startswith("Normal") else 1 for idx in ex_X.index], index=ex_X.index)

# save objects for later
joblib.dump(ex_X,"input/ex_X.pkl")
joblib.dump(ex_Y, "input/ex_Y.pkl")
joblib.dump(ex_groups,"input/ex_groups.pkl")
