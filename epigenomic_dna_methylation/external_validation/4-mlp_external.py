# MLP - EXTERNAL VALIDATION 

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
