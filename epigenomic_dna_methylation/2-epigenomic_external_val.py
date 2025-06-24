######### Load the external data 

# import necessary libraries
import pyreadr
import pandas as pd
import joblib

data = pyreadr.read_r("../input/external_top20_bVals.rds") # read in rds file 
df = data[None] # methylation beta value matrix
print(df.shape)
df.head()

# transpose the dataframe
df_t = df.T
print(df_t.shape)
df_t.head()

# load the metadata
metadata = pd.read_csv("../input/methylation2_sample_sheet.csv")
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
joblib.dump(X,"../input/ex_X.pkl")
joblib.dump(Y, "../input/ex_Y.pkl")
joblib.dump(groups,"../input/ex_groups.pkl")



######### External validation

# import necessary libaries
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from external_val_function import external_val

#load objects
X_train = joblib.load("../input/X.pkl")
Y_train  = joblib.load("../input/Y.pkl")

X_test= joblib.load("../input/ex_X.pkl")
Y_test  = joblib.load("../input/ex_Y.pkl")


### Support Vector Machine (SVM)

print("Support vector machine classifiers - external validation\n")

svm_tts_model = joblib.load("../input/svm_tts_model.pkl")
svm_bs_model = joblib.load("../input/svm_bs_model.pkl") 
svm_kcv_model = joblib.load("../input/svm_kcv_model.pkl") 
svm_loocv_model = joblib.load("../input/svm_loocv_model.pkl") 

# Train-test split model
external_val(model=svm_tts_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="Train-test split",
             roc_curve_path="results/svm_tts_ex_roc_curve.png",
             refit=False)

# Bootstrapping model
external_val(model=svm_bs_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="Bootstrapping",
             roc_curve_path="results/svm_bs_roc_curve.png",
             refit=True)

# K-fold CV model
external_val(model=svm_kcv_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="5-fold CV",
             roc_curve_path="results/svm_kcv_roc_curve.png",
             refit=True)

# LOGO-CV model
external_val(model=svm_loocv_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="LOGO-CV",
             roc_curve_path="results/svm_logocv_roc_curve.png",
             refit=True)


### Random Forest (RF)

print("Random forest classifiers - external validation\n")

rf_tts_model = joblib.load("../input/rf_tts_model.pkl")
rf_bs_model = joblib.load("../input/rf_bs_model.pkl") 
rf_kcv_model = joblib.load("../input/rf_kcv_model.pkl") 
rf_loocv_model = joblib.load("../input/rf_loocv_model.pkl") 

# Train-test split model
external_val(model=rf_tts_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="Train-test split",
             roc_curve_path="results/rf_tts_ex_roc_curve.png",
             refit=False)

# Bootstrapping model
external_val(model=rf_bs_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="Bootstrapping",
             roc_curve_path="results/rf_bs_roc_curve.png",
             refit=True)

# K-fold CV model
external_val(model=rf_kcv_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="5-fold CV",
             roc_curve_path="results/rf_kcv_roc_curve.png",
             refit=True)

# LOGO-CV model
external_val(model=rf_loocv_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="LOGO-CV",
             roc_curve_path="results/rf_logocv_roc_curve.png",
             refit=True)


### Multilayer Perceptron (MLP)

print("Multilayer perceptron classifiers - external validation\n")

mlp_tts_model = joblib.load("../input/mlp_tts_model.pkl")
mlp_bs_model = joblib.load("../input/mlp_bs_model.pkl") 
mlp_kcv_model = joblib.load("../input/mlp_kcv_model.pkl") 
mlp_loocv_model = joblib.load("../input/mlp_loocv_model.pkl") 

# Train-test split model
external_val(model=mlp_tts_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="Train-test split",
             roc_curve_path="results/mlp_tts_ex_roc_curve.png",
             refit=False)

# Bootstrapping model
external_val(model=mlp_bs_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="Bootstrapping",
             roc_curve_path="results/mlp_bs_roc_curve.png",
             refit=True)

# K-fold CV model
external_val(model=mlp_kcv_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="5-fold CV",
             roc_curve_path="results/mlp_kcv_roc_curve.png",
             refit=True)

# LOGO-CV model
external_val(model=mlp_loocv_model,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test,
             model_name="LOGO-CV",
             roc_curve_path="results/mlp_logocv_roc_curve.png",
             refit=True)
