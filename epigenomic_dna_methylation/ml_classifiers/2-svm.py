### Support vector machine (SVM) classifier 

# import necessary libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold, LeaveOneGroupOut 
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
 
# load objects
X = joblib.load("input/X.pkl")
Y  = joblib.load("input/Y.pkl")
groups = joblib.load("input/groups.pkl")

# import functions
from internal_val_functions import train_test_val, bootstrap_val, kfold_val, loocv_val


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
