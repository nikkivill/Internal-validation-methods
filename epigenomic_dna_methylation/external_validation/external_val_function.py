# EXTERNAL VALIDATION FUNCTION  

# import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve


def external_val(model, X_train, Y_train, X_test, Y_test,
                        model_name, roc_curve_path, refit=False):
    
    """
    Applies external validation for a model, giving the option to refit on all of the data used for development. 

    Parameters:
    - model: classification model obtained from internal validation (SVM, RF, MLP)
    - X_train, Y_train: data used for model development
    - X_test, Y_test: external test data 
    - model_name: name of the internal validation method for the ROC curve legend and plot title
    - roc_curve_path: path to save the ROC curve PDF
    - refit: if True, refit the model on full training data

    Returns:
    - metrics: performance metrics for external validation 
    """

    # subset X_test to features in X_train
    X_test = X_test[X_train.columns]

    # if refit is True
    if refit:
        print("Refitting model on entire training data")
        model.fit(X_train, Y_train)

    # predict on external data
    Y_pred = model.predict(X_test)
    Y_prob = model.predict_proba(X_test)[:, 1]

    # calculate performance metrics
    auc = roc_auc_score(Y_test, Y_prob)
    acc = accuracy_score(Y_test, Y_pred)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp +fp)
    f1 = 2 * (prec * sens) / (prec + sens)
    
    metrics = {
        'AUC': round(auc, 3),
        'Accuracy': round(acc, 3),
        'Sensitivity': round(sens, 3),
        'Specificity': round(spec, 3),
        'Precision': round(prec, 3),
        'F1': round(f1, 3)
        }
    
    print(f"{model_name} performance metrics:")
    for name, val in metrics.items():
        print(f"  {name}: {val}")

    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix:")
    print(cm)

     # compute tpr and fpr for roc curve
    fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
    # plot roc curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} - (AUC = {auc:.3f})')
    plt.plot([0,1], [0,1], 'r--', label=f'Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate - FPR')
    plt.ylabel('True Positive Rate - TPR')
    plt.title(f'External validation - {model_name} on epigenomic data')
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True)
    
    # save roc curve as PDF
    plt.savefig(roc_curve_path, bbox_inches='tight')
    plt.close()   
