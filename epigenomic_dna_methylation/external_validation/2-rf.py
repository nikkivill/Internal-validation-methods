### Random Forest (RF)

# import necessary libaries
import joblib
import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)

# import function
from external_val_functions import external_val, plot_roc_curves 

# load objects
X_train = joblib.load("../input/X.pkl")
Y_train  = joblib.load("../input/Y.pkl")

X_test= joblib.load("../input/ex_X.pkl")
Y_test  = joblib.load("../input/ex_Y.pkl")


print("Random forest classifiers - external validation\n")

rf_tts_model = joblib.load("../input/rf_tts_model.pkl")
rf_tts_threshold = joblib.load("../input/rf_tts_threshold.pkl")

rf_bs_model = joblib.load("../input/rf_bs_model.pkl") 
rf_bs_threshold = joblib.load("../input/rf_bs_threshold.pkl")

rf_kcv_model = joblib.load("../input/rf_kcv_model.pkl") 
rf_kcv_threshold = joblib.load("../input/rf_kcv_threshold.pkl")

rf_logocv_model = joblib.load("../input/rf_logocv_model.pkl") 
rf_logocv_threshold = joblib.load("../input/rf_logocv_threshold.pkl")

# save roc data
roc_data = []

# Train-test split model
tts_metrics, tts_fpr, tts_tpr, tts_auc = external_val(model=rf_tts_model,
             threshold = rf_tts_threshold,                                         
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "RF - Train-test split",
             fit = False)

roc_data.append((tts_fpr, tts_tpr, tts_auc, "Train-test split"))

# Bootstrapping model
bs_metrics, bs_fpr, bs_tpr, bs_auc = external_val(model=rf_bs_model,
             threshold = rf_bs_threshold,
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "RF - Bootstrapping",
             fit = True)

roc_data.append((bs_fpr, bs_tpr, bs_auc, "Bootstrapping"))

# K-fold CV model
kcv_metrics, kcv_fpr, kcv_tpr, kcv_auc = external_val(model=rf_kcv_model,
             threshold = rf_kcv_threshold,                                          
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "RF - 5-fold CV",
             fit = True)

roc_data.append((kcv_fpr, kcv_tpr, kcv_auc, "5-fold CV"))

# LOGO-CV model
logocv_metrics, logocv_fpr, logocv_tpr, logocv_auc = external_val(model=rf_logocv_model,
             threshold = rf_logocv_threshold,                                                      
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "RF - LOGOCV",
             fit = True)

roc_data.append((logocv_fpr, logocv_tpr, logocv_auc, "LOGOCV"))

# Plot all ROC curves together
plot_roc_curves(
    roc_data=roc_data,
    plot_title="RF - External Validation ROC Curves",
    save_path="results/rf_all_val_roc_curve.png"
)
