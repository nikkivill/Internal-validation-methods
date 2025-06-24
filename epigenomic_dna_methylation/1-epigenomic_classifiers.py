######## Load the data

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


######### Machine learning classifiers

# import necessary libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold, LeaveOneGroupOut 
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

# load objects
X = joblib.load("input/X.pkl")
Y  = joblib.load("input/Y.pkl")
groups = joblib.load("input/groups.pkl")

# import functions
from internal_val_functions import train_test_val, bootstrap_val, kfold_val, loocv_val


### Support vector machine (SVM) classifier 

### Run train-test split for SVM 
train_test_val(X = X,
               Y = Y,
               groups = groups,
               classifier= SVC(kernel='rbf', probability=True, random_state=0), 
               param_grid= {'C': [0.1, 1, 10, 100, 1000], 
                            'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
               roc_curve_classifier= "Support Vector Machine (SVM)",
               model_path= "input/svm_tts_model.pkl",
               roc_curve_path= "results/svm_tts_roc_curve.pdf",
               r_state=0)

### Run bootstrapping for SVM 
bootstrap_val(X = X,
              Y = Y,
              groups = groups,
              classifier= SVC(kernel='rbf', probability=True, random_state=0), 
              param_grid= {'C': [0.1, 1, 10, 100, 1000],
                           'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
              model_path= "input/svm_bs_model.pkl",
              n_bootstraps=100,
              r_state=0)

### Run k-fold cross-validation for SVM 
kfold_val(X = X,
          Y = Y,
          groups = groups,
          classifier= SVC(kernel='rbf', probability=True, random_state=0), 
          param_grid= {'C': [0.1, 1, 10, 100, 1000],
                       'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
          model_path= "input/svm_kcv_model.pkl",
          k=5,
          r_state=0)

### Run LOOCV for SVM 
loocv_val(X = X,
          Y = Y,
          groups = groups,
          classifier= SVC(kernel='rbf', probability=True, random_state=0), 
          param_grid= {'C': [0.1, 1, 10, 100, 1000], 
                       'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
          model_path= "input/svm_loocv_model.pkl",
          r_state=0)


### Random Forest (RF) classifier

### Run train-test split for RF
train_test_val(X = X,
               Y = Y,
               groups = groups,
               classifier= RandomForestClassifier(random_state=0), 
               param_grid= [{'n_estimators': [100, 200, 500, 1000],
                             'max_depth': [None, 5, 10, 30],
                             'max_features': ['sqrt', 'log2'],
                             'min_samples_split': [2, 5, 10],
                             'min_samples_leaf': [1, 2, 5, 10]}],
               roc_curve_classifier= "Random Forest (RF)",
               model_path= "input/rf_tts_model.pkl",
               roc_curve_path= "results/rf_tts_roc_curve.pdf",
               r_state=0)

### Run bootstrapping for RF
bootstrap_val(X = X,
              Y = Y,
              groups = groups,
              classifier= RandomForestClassifier(random_state=0),
              param_grid= [{'n_estimators': [100, 200, 500, 1000],
                             'max_depth': [None, 5, 10, 30],
                             'max_features': ['sqrt', 'log2'],
                             'min_samples_split': [2, 5, 10],
                             'min_samples_leaf': [1, 2, 5, 10]}],
              model_path= "input/rf_bs_model.pkl",
              n_bootstraps=100,
              r_state=0)

### Run k-fold cross-validation for RF
kfold_val(X = X,
          Y = Y,
          groups = groups,
          classifier= RandomForestClassifier(random_state=0),
          param_grid= [{'n_estimators': [100, 200, 500, 1000],
                        'max_depth': [None, 5, 10, 30],
                        'max_features': ['sqrt', 'log2'],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 5, 10]}],
          model_path= "input/rf_kcv_model.pkl",
          k=5,
          r_state=0)

### Run LOOCV for RF
loocv_val(X = X,
          Y = Y,
          groups = groups,
          classifier= RandomForestClassifier(random_state=0),
          param_grid= [{'n_estimators': [100, 200, 500, 1000],
                        'max_depth': [None, 5, 10, 30],
                        'max_features': ['sqrt', 'log2'],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 5, 10]}],
          model_path= "input/rf_loocv_model.pkl",
          r_state=0)


### Artificial neural network - MLP (multilayer perceptron) classifier

### Run train-test split for MLP
train_test_val(X = X,
               Y = Y,
               groups = groups,
               classifier= MLPClassifier(random_state=0, max_iter=2000),
               param_grid= [{'hidden_layer_sizes':[(10,), (15,), (18,), (10,15), (18,10)],
                             'alpha':[0.0001, 0.001, 0.01, 0.1, 1],
                             'learning_rate_init':[0.1, 0.01, 0.001]}],
               roc_curve_classifier= "Multilayer perceptron (MLP)",
               model_path= "input/mlp_tts_model.pkl",
               roc_curve_path= "results/mlp_tts_roc_curve.pdf",
               r_state=0)

### Run bootstrapping for MLP
bootstrap_val(X = X,
              Y = Y,
              groups = groups,
              classifier= MLPClassifier(random_state=0, max_iter=2000),
              param_grid= [{'hidden_layer_sizes':[(10,), (15,), (18,), (10,15), (18,10)],
                             'alpha':[0.0001, 0.001, 0.01, 0.1, 1],
                             'learning_rate_init':[0.1, 0.01, 0.001]}],
              model_path= "input/mlp_bs_model.pkl",
              n_bootstraps=100,
              r_state=0)

### Run k-fold cross-validation for MLP
kfold_val(X = X,
          Y = Y,
          groups = groups,
          classifier= MLPClassifier(random_state=0, max_iter=2000),
          param_grid= [{'hidden_layer_sizes':[(10,), (15,), (18,), (10,15), (18,10)],
                        'alpha':[0.0001, 0.001, 0.01, 0.1, 1],
                        'learning_rate_init':[0.1, 0.01, 0.001]}],
          model_path= "input/mlp_kcv_model.pkl",
          k=5,
          r_state=0)

### Run LOOCV for MLP
loocv_val(X = X,
          Y = Y,
          groups = groups,
          classifier= MLPClassifier(random_state=0, max_iter=2000),
          param_grid= [{'hidden_layer_sizes':[(10,), (15,), (18,), (10,15), (18,10)],
                        'alpha':[0.0001, 0.001, 0.01, 0.1, 1],
                        'learning_rate_init':[0.1, 0.01, 0.001]}],
          model_path= "input/mlp_loocv_model.pkl",
          r_state=0)
