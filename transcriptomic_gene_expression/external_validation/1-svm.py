### Support Vector Machine (SVM)

# import necessary libraries
import joblib
import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

# import function
from external_val_functions import external_val, plot_roc_curves

#load objects
X_train = joblib.load("../input/X.pkl")
Y_train  = joblib.load("../input/Y.pkl")

X_test= joblib.load("../input/ex_X.pkl")
Y_test  = joblib.load("../input/ex_Y.pkl")


print("Support vector machine classifiers - external validation\n")

svm_tts_model = joblib.load("../input/svm_tts_model.pkl")
svm_tts_threshold = joblib.load("../input/svm_tts_threshold.pkl")
svm_tts_scaler = joblib.load("../input/svm_tts_scaler.pkl")

svm_bs_model = joblib.load("../input/svm_bs_model.pkl") 
svm_bs_threshold  = joblib.load("../input/svm_bs_threshold.pkl")

svm_kcv_model = joblib.load("../input/svm_kcv_model.pkl") 
svm_kcv_threshold = joblib.load("../input/svm_kcv_threshold.pkl")

svm_logocv_model = joblib.load("../input/svm_logocv_model.pkl") 
svm_logocv_threshold = joblib.load("../input/svm_logocv_threshold.pkl")

# save roc data
roc_data = []

# Train-test split model
tts_metrics, tts_fpr, tts_tpr, tts_auc = external_val(model = svm_tts_model,
             threshold = svm_tts_threshold,
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "SVM - Train-test split",
             fit = False,
             scaler = svm_tts_scaler)

roc_data.append((tts_fpr, tts_tpr, tts_auc, "Train-test split"))

# Bootstrapping model
bs_metrics, bs_fpr, bs_tpr, bs_auc = external_val(model = svm_bs_model,
             threshold = svm_bs_threshold,
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "SVM - Bootstrapping",
             fit = True)

roc_data.append((bs_fpr, bs_tpr, bs_auc, "Bootstrapping"))

# K-fold CV model
kcv_metrics, kcv_fpr, kcv_tpr, kcv_auc = external_val(model = svm_kcv_model,
             threshold = svm_kcv_threshold,
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "SVM - 5-fold CV",
             fit = True)

roc_data.append((kcv_fpr, kcv_tpr, kcv_auc, "5-fold CV"))

# LOGO-CV model
logocv_metrics, logocv_fpr, logocv_tpr, logocv_auc = external_val(model = svm_logocv_model,
             threshold = svm_logocv_threshold,
             X_train = X_train,
             Y_train = Y_train,
             X_test = X_test,
             Y_test = Y_test,
             clf_method = "SVM - LOGOCV",
             fit = True)

roc_data.append((logocv_fpr, logocv_tpr, logocv_auc, "LOGOCV"))


# Plot all ROC curves together
plot_roc_curves(
    roc_data=roc_data,
    plot_title="SVM - External Validation ROC Curves",
    save_path="results/svm_all_val_roc_curve.png"
)
