######### 1 - load the data 

# import necessary libraries
import pyreadr
import pandas as pd
import joblib

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
metadata = metadata.loc[df.columns]

print(metadata.shape)
metadata.head()

# group paired samples together using metadata
# using original df where columns=samples
sample_ids = df.columns
# look up sample_source values for sample_ids
groups = metadata.loc[sample_ids, 'Sample_Source'].values 
print(groups)

### specify features and labels
X = df.T
# create binary labels with 0 = Normal, 1 = Tumor 
# convert to pandas.Series to use .iloc later
Y = pd.Series([0 if col.startswith("Normal") else 1 for col in df.columns], index=df.columns, name='label')

# save objects for later
joblib.dump(X,"input/X.pkl")
joblib.dump(Y, "input/Y.pkl")
joblib.dump(groups,"input/groups.pkl")
