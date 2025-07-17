### EXTERNAL VALIDATION FUNCTIONS

# import necessary libraries
import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)


def external_val(model, threshold,
                 X_train, Y_train, X_test, Y_test,
                 clf_method, fit=False, scaler=None):
    
    """
    Applies external validation for a model, giving the option to refit on all of the data used for development. 

    Parameters:
    - model (sklearn estimator): Optimised machine learning classifier obtained from internal validation 
    - threshold (float): Decision threshold for predictions (convert predicted probabilities to class predictions)
    - X_train (pd.DataFrame): Training features 
    - Y_train (pd.Series): Training labels
    - X_test (pd.DataFrame): External test features
    - Y_test (pd.Series): External test labels
    - clf_method (str): Name of classifier and internal validation method used
    - fit (bool, optional): If True, fit the model on full training data
    - scaler (sklearn.base.TransformerMixin, optional): Scaler used to transform external test features when 'fit=False'

    Returns:
    - metrics: Dictionary of performance metrics on the test set, including:
               - 'AUC'
               - 'Accuracy'
               - 'Sensitivity'
               - 'Specificity'
               - 'Precision'
               - 'F1'
    """

    # ensure scaler is provided if fit is not true
    if not fit and scaler is None:
        raise ValueError("Scaler must be provided if fit is False.")

   # if fit is true
    if fit:
        print("Fitting scaler and model on entire training data")

        # fit scaler on all the training data 
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train),
                               columns=X_train.columns,
                               index=X_train.index)
        # scale external test data using the same scaler
        X_test = pd.DataFrame(scaler.transform(X_test),
                              columns=X_test.columns,
                              index=X_test.index)
        
        model.fit(X_train, Y_train)

    else:
        # scale external test data using the scaler input
        print("Using provided pre-fitted scaler and model")
        X_test = pd.DataFrame(scaler.transform(X_test),
                              columns=X_test.columns,
                              index=X_test.index)
    
    # evaluate on test set using best model and threshold
    Y_prob = model.predict_proba(X_test)[:, 1]
    Y_pred = (Y_prob >= threshold).astype(int)
        
    # calculate performance metrics
    auc = roc_auc_score(Y_test, Y_prob)
    acc = accuracy_score(Y_test, Y_pred)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    sens = recall_score(Y_test, Y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) != 0 else 0
    prec = precision_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)
    
    metrics = {
        'AUC': round(auc, 3),
        'Accuracy': round(acc, 3),
        'Sensitivity': round(sens, 3),
        'Specificity': round(spec, 3),
        'Precision': round(prec, 3),
        'F1': round(f1, 3)
        }
    
    # print results
    print(model)
    print(threshold)
    print(f"{clf_method} performance metrics:")
    for name, val in metrics.items():
        print(f"  {name}: {val}")

    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix:")
    print(cm)

    # compute tpr and fpr for roc curve
    fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)

    return metrics, fpr, tpr, auc 


def plot_roc_curves(roc_data, plot_title, save_path):

    """
    Plots multiple ROC curves on a single graph.

    Parameters:
    - roc_data (list): A list of tuples which contain fpr, tpr, auc, and label name for internal validation method
    - plot_title (str): The main title for the ROC curve plot
    - save_path (str): Full file path to save the ROC curve

    """

    plt.figure(figsize=(8,7))

    # plot each ROC curve for each internal validation method
    for fpr, tpr, auc, label_name in roc_data:
        plt.plot(fpr, tpr, label=f"{label_name} (AUC) = {auc:.3f}")

    plt.plot([0,1], [0,1], 'r--', label=f'Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate - FPR')
    plt.ylabel('True Positive Rate - TPR')
    plt.title(plot_title)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)

    # save roc curve 
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close() 
