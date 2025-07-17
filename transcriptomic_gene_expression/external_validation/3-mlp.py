### Multilayer Perceptron (MLP)

# import necessary libraries
import joblib
import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)

# import function
from external_val_functions import external_val, plot_roc_curves

#load objects
X_train = joblib.load("../input/X.pkl")
Y_train  = joblib.load("../input/Y.pkl")

X_test= joblib.load("../input/ex_X.pkl")
Y_test  = joblib.load("../input/ex_Y.pkl")

print("Multilayer perceptron classifiers - external validation\n")

mlp_tts_model = joblib.load("../input/mlp_tts_model.pkl")
mlp_tts_threshold = joblib.load("../input/mlp_tts_threshold.pkl")
mlp_tts_scaler = joblib.load("../input/mlp_tts_scaler.pkl")

mlp_bs_model = joblib.load("../input/mlp_bs_model.pkl") 
mlp_bs_threshold  = joblib.load("../input/mlp_bs_threshold.pkl")

mlp_kcv_model = joblib.load("../input/mlp_kcv_model.pkl") 
mlp_kcv_threshold = joblib.load("../input/mlp_kcv_threshold.pkl")

mlp_logocv_model = joblib.load("../input/mlp_logocv_model.pkl") 
mlp_logocv_threshold = joblib.load("../input/mlp_logocv_threshold.pkl")

# save roc data
roc_data = []

# Train-test split model
tts_metrics, tts_fpr, tts_tpr, tts_auc = external_val(model = mlp_tts_model,
             threshold = mlp_tts_threshold,
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "MLP - Train-test split",
             fit = False,
             scaler = mlp_tts_scaler)

roc_data.append((tts_fpr, tts_tpr, tts_auc, "Train-test split"))

# Bootstrapping model
bs_metrics, bs_fpr, bs_tpr, bs_auc = external_val(model = mlp_bs_model,
             threshold = mlp_bs_threshold,
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "MLP - Bootstrapping",
             fit = True)

roc_data.append((bs_fpr, bs_tpr, bs_auc, "Bootstrapping"))

# K-fold CV model
kcv_metrics, kcv_fpr, kcv_tpr, kcv_auc = external_val(model = mlp_kcv_model,
             threshold = mlp_kcv_threshold,
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "MLP - 5-fold CV",
             fit = True)

roc_data.append((kcv_fpr, kcv_tpr, kcv_auc, "5-fold CV"))

# LOGO-CV model
logocv_metrics, logocv_fpr, logocv_tpr, logocv_auc = external_val(model = mlp_logocv_model,
             threshold = mlp_logocv_threshold,
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "MLP - LOGOCV",
             fit = True)

roc_data.append((logocv_fpr, logocv_tpr, logocv_auc, "LOGOCV"))


# Plot all ROC curves together
plot_roc_curves(
    roc_data=roc_data,
    plot_title="MLP - External Validation ROC Curves",
    save_path="results/mlp_all_val_roc_curve.png"
)
