######## KNN classifier

# import necessary libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold, LeaveOneGroupOut 
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

# load objects
X = joblib.load("input/X.pkl")
Y  = joblib.load("input/Y.pkl")
groups = joblib.load("input/groups.pkl")

# import functions
from internal_val_functions import train_test_val, bootstrap_val, kfold_val, logocv_val


### Run train-test split for KNN
train_test_val(X = X,
               Y = Y,
               groups = groups,
               classifier = KNeighborsClassifier(),
               param_grid = [{'n_neighbors': list(range(1,31))}],
               roc_curve_classifier = "KNN",
               scaler_path = "input/knn_tts_scaler.pkl",
               model_path = "input/knn_tts_model.pkl",
               threshold_path = "input/knn_tts_threshold.pkl",
               roc_curve_path = "results/knn_tts_roc_curve.png",
               r_state = 0)

### Run bootstrapping for KNN
bootstrap_val(X = X,
              Y = Y,
              groups = groups,
              classifier = KNeighborsClassifier(),
              param_grid = [{'n_neighbors': list(range(1,31))}],
              model_path = "input/knn_bs_model.pkl",
              threshold_path = "input/knn_bs_threshold.pkl",
              n_bootstraps = 100,
              r_state = 0)

### Run k-fold cross-validation for KNN
kfold_val(X = X,
          Y = Y,
          groups = groups,
          classifier = KNeighborsClassifier(),
          param_grid = [{'n_neighbors': list(range(1,31))}],
          model_path = "input/knn_kcv_model.pkl",
          threshold_path = "input/knn_kcv_threshold.pkl",
          k = 5,
          r_state = 0)

### Run LOOCV for KNN
logocv_val(X = X,
          Y = Y,
          groups = groups,
          classifier = KNeighborsClassifier(),
          param_grid = [{'n_neighbors': list(range(1,31))}],
          model_path = "input/knn_logocv_model.pkl",
          threshold_path = "input/knn_logocv_threshold.pkl",
          r_state = 0)
