### Artificial neural network - MLP (multilayer perceptron) classifier

# import necessary libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
