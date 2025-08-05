######## RF classifier

# import necessary libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold, LeaveOneGroupOut 
from sklearn.utils import resample
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)

# load objects
X = joblib.load("input/X.pkl")
Y  = joblib.load("input/Y.pkl")
groups = joblib.load("input/groups.pkl")

# import functions
from internal_val_functions import train_test_val, bootstrap_val, kfold_val, logocv_val


### Random Forest (RF) classifier

### Run train-test split for RF
train_test_val(X = X,
               Y = Y,
               groups = groups,
               classifier = RandomForestClassifier(random_state=0), 
               param_grid = [{'n_estimators': [100, 200, 500, 1000],
                             'max_depth': [None, 5, 10],
                             'max_features': ['sqrt', 'log2'],
                             'min_samples_leaf': [1, 0.01, 0.10]}],
               roc_curve_classifier = "RF",
               scaler_path = "input/rf_tts_scaler.pkl",
               model_path = "input/rf_tts_model.pkl",
               threshold_path = "input/rf_tts_threshold.pkl",
               roc_curve_path = "results/rf_tts_roc_curve.png",
               r_state = 0)

### Run bootstrapping for RF
bootstrap_val(X = X,
              Y = Y,
              groups = groups,
              classifier = RandomForestClassifier(random_state=0),
              param_grid = [{'n_estimators': [100, 200, 500, 1000],
                             'max_depth': [None, 5, 10],
                             'max_features': ['sqrt', 'log2'],
                             'min_samples_leaf': [1, 0.01, 0.10]}],
              model_path = "input/rf_bs_model.pkl",
              threshold_path = "input/rf_bs_threshold.pkl",
              n_bootstraps = 100,
              r_state = 0)

### Run k-fold cross-validation for RF
kfold_val(X = X,
          Y = Y,
          groups = groups,
          classifier = RandomForestClassifier(random_state=0),
          param_grid = [{'n_estimators': [100, 200, 500, 1000],
                         'max_depth': [None, 5, 10],
                         'max_features': ['sqrt', 'log2'],
                         'min_samples_leaf': [1, 0.01, 0.10]}],
          model_path = "input/rf_kcv_model.pkl",
          threshold_path = "input/rf_kcv_threshold.pkl",
          k = 5,
          r_state = 0)

### Run LOGOCV for RF
logocv_val(X = X,
          Y = Y,
          groups = groups,
          classifier = RandomForestClassifier(random_state=0),
          param_grid = [{'n_estimators': [100, 200, 500, 1000],
                         'max_depth': [None, 5, 10],
                         'max_features': ['sqrt', 'log2'],
                         'min_samples_leaf': [1, 0.01, 0.10]}],
          model_path = "input/rf_logocv_model.pkl",
          threshold_path = "input/rf_logocv_threshold.pkl",
          r_state = 0)
