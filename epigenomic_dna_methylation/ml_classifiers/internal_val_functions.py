### INTERNAL VALIDATION FUNCTIONS 


# import necessary libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold, LeaveOneGroupOut 
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve


### Train-test split function

def train_test_val(X, Y, groups, classifier, param_grid, roc_curve_classifier,
                   model_path, roc_curve_path, r_state=0):
    
    """
    Applies train-test split (80/20) validation for a classifier.

    Parameters:
    - X: features (pandas dataframe)
    - Y: class labels (series)
    - groups: group labels (for paired splitting)
    - classifier: machine learning classifier 
    - param_grid: dict or list of dicts for classifier hyperparameter tuning using GridSearchCV
    - roc_curve_title: name of classifier (e.g. Random Forest)
    - model_path: full path to save the final model (.pkl)
    - roc_path: full path to save the ROC curve (.pdf)
    - r_state: random seed

    Returns:
    - best_model: classifier model with tuned hyperparameters
    - metrics: dictionary of performance metrics
    """

    # split the data by groups as samples are paired - prevents data leakage 
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=r_state)
    train_index, test_index = next(gss.split(X, Y, groups=groups))
    # apply the split to data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    # group labels from training set to keep the same samples in the same fold
    groups_train = groups[train_index]

    # define cv for GridSearchCV
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=r_state)

    # hyperparameter tuning for classifier using param_grid
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1)

    grid_search.fit(X_train, Y_train, groups=groups_train)
    # best model
    tts_model = grid_search.best_estimator_

    # evaluate on test set
    Y_pred = tts_model.predict(X_test)
    Y_prob = tts_model.predict_proba(X_test)[:, 1]

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

    # compute tpr and fpr for roc curve
    fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
    # plot roc curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'{roc_curve_classifier} - train-test split (AUC = {auc:.3f})')
    plt.plot([0,1], [0,1], 'r--', label=f'Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate - FPR')
    plt.ylabel('True Positive Rate - TPR')
    plt.title(f'ROC Curve - {roc_curve_classifier} (train-test split) on epigenomic data')
    plt.legend(loc='lower right')
    plt.grid(True)

    # save roc curve as PDF
    plt.savefig(roc_curve_path, bbox_inches='tight')
    plt.close()   
    
    # save train-test split model
    joblib.dump(tts_model, model_path)

    print(tts_model)
    print("Train-test split performance metrics:")
    for name, val in metrics.items():
        print(f"  {name}: {val}")
    


### Bootstrapping function

def bootstrap_val(X, Y, groups, classifier, param_grid, model_path,
                  n_bootstraps=100, r_state=0):
    
    """
    Applies bootstrap (including .632 and .632+) validation for a classifier.

    Parameters:
    - X: features (pandas dataframe)
    - Y: class labels (series)
    - groups: group labels (for paired splitting)
    - classifier: machine learning classifier 
    - param_grid: dict or list of dicts for classifier hyperparameter tuning using GridSearchCV
    - model_path: full path to save the final model (.pkl)
    - n_bootstraps: number of bootstrap iterations 
    - r_state: random seed

    Returns:
    - best_model: classifier model with tuned hyperparameters
    - metrics: dictionary of performance metrics for bootstrap, .632 and .632+ (including confidence intervals)
    """

    # groups for splitting
    group_ids = pd.Series(groups, index=X.index)
    unique_groups = group_ids.unique()
    
    # initalise performance metrics lists
    bs_auc = [] 
    bs_acc = [] 
    bs_sens = []
    bs_spec = []
    bs_prec = []
    bs_f1 = []

    bs_auc_632 = []
    bs_acc_632 = [] 
    bs_sens_632 = []
    bs_spec_632 = []
    bs_prec_632 = []
    bs_f1_632 = []

    bs_auc_632_plus = []
    bs_acc_632_plus = []
    bs_sens_632_plus = []
    bs_spec_632_plus = []
    bs_prec_632_plus = []
    bs_f1_632_plus = []  
    
    # initalise selected hyperparameter list
    selected_params = []

    for i in range(n_bootstraps): 
       print(f"Bootstrap iteration {i+1}/{n_bootstraps}", flush=True) 

       # split data into training and testing by resampling with replacement (keep groups together)
       sampled_groups = resample(
          unique_groups,
          replace=True,
          n_samples=len(unique_groups),
          random_state=i) # ensures randomness in splits per iteration and reproducibility
       
       # out-of-bag (OOB) testing groups
       oob_groups = np.setdiff1d(unique_groups, sampled_groups)
       # mask to select the samples for training and testing
       train_mask = group_ids.isin(sampled_groups)
       oob_mask = group_ids.isin(oob_groups)
       
       # if no OOB set, skip bootstrap 
       if not oob_mask.any():
          continue
       # subset data to training and testing
       X_train = X[train_mask]
       Y_train = Y[train_mask]
       X_oob = X[oob_mask]
       Y_oob = Y[oob_mask]
       groups_train = group_ids[train_mask]
       
       # define cv for GridSearchCV
       cv = GroupKFold(n_splits=5, shuffle=True, random_state=r_state)
       
       # hyperparameter tuning for classifier using param_grid
       grid_search = GridSearchCV(
           estimator=classifier,
           param_grid=param_grid,
           cv=cv,
           scoring='roc_auc',
           n_jobs=-1) 
       
       grid_search.fit(X_train, Y_train, groups=groups_train)
       
       # selected parameters
       selected_params.append(grid_search.best_params_)
       # best model
       best_model = grid_search.best_estimator_
       
       # predictions on OOB set
       Y_pred = best_model.predict(X_oob)
       Y_prob = best_model.predict_proba(X_oob)[:, 1]
       
       # calculate performance metrics 
       auc = roc_auc_score(Y_oob, Y_prob)
       acc = accuracy_score(Y_oob, Y_pred)
       tn, fp, fn, tp = confusion_matrix(Y_oob, Y_pred).ravel()
       sens = tp / (tp + fn)
       spec = tn / (tn + fp)
       prec = tp / (tp +fp)
       f1 = 2 * (prec * sens) / (prec + sens)
       
       # append performance metrics
       bs_auc.append(auc)
       bs_acc.append(acc)
       bs_sens.append(sens)
       bs_spec.append(spec)
       bs_prec.append(prec)
       bs_f1.append(f1)
       
       # prediction on training set for .632
       Y_train_pred = best_model.predict(X_train)
       Y_train_prob = best_model.predict_proba(X_train)[:, 1]
       
       # calculate training performance metrics for .632 and .632+
       train_auc = roc_auc_score(Y_train, Y_train_prob)
       train_acc = accuracy_score(Y_train, Y_train_pred)
       tn_t, fp_t, fn_t, tp_t = confusion_matrix(Y_train, Y_train_pred).ravel()
       train_sens = tp_t / (tp_t + fn_t)
       train_spec = tn_t / (tn_t + fp_t)
       train_prec = tp_t / (tp_t + fp_t)
       train_f1 = 2 * (train_prec * train_sens) / (train_prec + train_sens)
       
       # .632 performance metrics
       auc_632 = 0.368 * train_auc + 0.632 * auc
       acc_632 = 0.368 * train_acc + 0.632 * acc
       sens_632 = 0.368 * train_sens + 0.632 * sens
       spec_632 = 0.368 * train_spec + 0.632 * spec
       prec_632 = 0.368 * train_prec + 0.632 * prec
       f1_632 = 0.368 * train_f1 + 0.632 * f1
       
       # append .632 performance metrics
       bs_auc_632.append(auc_632)
       bs_acc_632.append(acc_632)
       bs_sens_632.append(sens_632)
       bs_spec_632.append(spec_632)
       bs_prec_632.append(prec_632)
       bs_f1_632.append(f1_632)

       # .632+ performance metrics
       metrics_train = {
           "auc": train_auc,
           "acc": train_acc,
           "sens": train_sens,
           "spec": train_spec,
           "prec": train_prec,
           "f1": train_f1
           }
       
       metrics_oob = {
           "auc": auc,
           "acc": acc,
           "sens": sens,
           "spec": spec,
           "prec": prec,
           "f1": f1
           }
       
       append_632_plus = {
           "auc": bs_auc_632_plus,
           "acc": bs_acc_632_plus,
           "sens": bs_sens_632_plus,
           "spec": bs_spec_632_plus,
           "prec": bs_prec_632_plus,
           "f1": bs_f1_632_plus
           }
       
       for key, lst in append_632_plus.items():
          train_val = metrics_train[key]
          oob_val = metrics_oob[key]
          
          if (1 - train_val) == 0:
             R = 0
          else:
             R = (1 - oob_val) / (1 - train_val)
             
          w = 0.632 / (1 - 0.368 * R)
          w = np.clip(w, 0, 1)
          
          metrics_632_plus = (1 - w) * train_val + w * oob_val
          lst.append(metrics_632_plus)
          
    # function to format mean ± std + 95% CI
    def format_metric(values):
       mean = np.mean(values)
       std = np.std(values)
       ci_lower = np.percentile(values, 2.5)
       ci_upper = np.percentile(values, 97.5)
       return f"{mean:.3f} ± {std:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])"
    
    bs_metrics = {
       "Mean AUC": format_metric(bs_auc),
       "Mean accuracy": format_metric(bs_acc),
       "Mean sensitivity": format_metric(bs_sens),
       "Mean specificity": format_metric(bs_spec),
       "Mean precision": format_metric(bs_prec),
       "Mean F1 score": format_metric(bs_f1)
       }
    
    bs_metrics_632 = {
       "Mean AUC (.632)": format_metric(bs_auc_632),
       "Mean accuracy (.632)": format_metric(bs_acc_632),
       "Mean sensitivity (.632)": format_metric(bs_sens_632),
       "Mean specificity (.632)": format_metric(bs_spec_632),
       "Mean precision (.632)": format_metric(bs_prec_632),
       "Mean F1 score (.632)": format_metric(bs_f1_632)
       }
    
    bs_metrics_632_plus = {
       "Mean AUC (.632+)": format_metric(bs_auc_632_plus),
       "Mean accuracy (.632+)": format_metric(bs_acc_632_plus),
       "Mean sensitivity (.632+)": format_metric(bs_sens_632_plus),
       "Mean specificity (.632+)": format_metric(bs_spec_632_plus),
       "Mean precision (.632+)": format_metric(bs_prec_632_plus),
       "Mean F1 score (.632+)": format_metric(bs_f1_632_plus)
       }
    
    # find the best model hyperparameter(s) based on the mode 
    final_params = pd.Series([tuple(sorted(p.items())) for p in selected_params]).mode()[0]
    final_params = dict(final_params)
    
    # build final model
    bs_model = classifier.set_params(**final_params)
    # save bootstrap model
    joblib.dump(bs_model, model_path)
    
    # print results
    print(bs_model)
    print("\nBootstrapping performance metrics:")
    for name, val in bs_metrics.items():
        print(f"  {name}: {val}")

    print("\n.632 performance metrics:")
    for name, val in bs_metrics_632.items():
        print(f"  {name}: {val}")

    print("\n.632+ performance metrics:")
    for name, val in bs_metrics_632_plus.items():
        print(f"  {name}: {val}")
  





### K-fold cross-validation function 

def kfold_val(X, Y, groups, classifier, param_grid,
              model_path, k=5, r_state=0):
    
    """
    Applies nested k-fold cross validation (outer=k, inner=5) for a classifier.

    Parameters:
    - X: features (pandas dataframe)
    - Y: class labels (series)
    - groups: group labels (for paired splitting)
    - classifier: machine learning classifier 
    - param_grid: dict or list of dicts for classifier hyperparameter tuning using GridSearchCV
    - model_path: full path to save the final model (.pkl)
    - k: number of folds for the outer cross-validation 
    - r_state: random seed

    Returns:
    - best_model: classifier model with tuned hyperparameters
    - metrics: dictionary of performance metrics for k-fold cross validation
    """

    # define cv split (5x5 nested group cv)
    outer_cv = GroupKFold(n_splits=k, shuffle=True, random_state=r_state)
    inner_cv = GroupKFold(n_splits=5, shuffle=True, random_state=r_state)
    
    # storing results
    outer_results = []
    selected_params = []

    # for each 'k' fold:
    for fold, (train_index, test_index) in enumerate(outer_cv.split(X, Y, groups), 1):
       
       # print which fold is currently being executed 
       print(f"Outer fold {fold}/{outer_cv.get_n_splits(groups=groups)}", flush=True)

       # split into the training and testing folds 
       X_train, X_test = X.iloc[train_index], X.iloc[test_index]
       Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
       groups_train = groups[train_index]

       # hyperparameter tuning for classifier using param_grid
       grid_search = GridSearchCV(
           estimator=classifier,
           param_grid=param_grid,
           cv=inner_cv,
           scoring='roc_auc',
           n_jobs=-1) 
       
       grid_search.fit(X_train, Y_train, groups=groups_train)
       
       # selected parameters
       selected_params.append(grid_search.best_params_)
       # best model
       best_model = grid_search.best_estimator_
       
       # test best model on the outer test fold
       Y_pred = best_model.predict(X_test)
       Y_prob = best_model.predict_proba(X_test)[:,1]
       
       # calculate performance metrics
       auc = roc_auc_score(Y_test, Y_prob)
       acc = accuracy_score(Y_test, Y_pred)
       tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
       sens = tp / (tp + fn)
       spec = tn / (tn + fp)
       prec = tp / (tp +fp)
       f1 = 2 * (prec * sens) / (prec + sens)
       
       outer_results.append({
          'auc': auc,
          'acc': acc,
          'sens': sens,
          'spec': spec,
          'prec': prec,
          'f1': f1
          })
    
    # extract results 
    kcv_auc  = [r['auc']  for r in outer_results]
    kcv_acc  = [r['acc']  for r in outer_results]
    kcv_sens = [r['sens'] for r in outer_results]
    kcv_spec = [r['spec'] for r in outer_results]
    kcv_prec = [r['prec'] for r in outer_results]
    kcv_f1 = [r['f1'] for r in outer_results]
    
    # function to format mean ± std + 95% CI
    def format_metric(values):
       mean = np.mean(values)
       std = np.std(values)
       ci_lower = np.percentile(values, 2.5)
       ci_upper = np.percentile(values, 97.5)
       return f"{mean:.3f} ± {std:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])"
    
    kcv_metrics = {
       "Mean AUC": format_metric(kcv_auc),
       "Mean accuracy": format_metric(kcv_acc),
       "Mean sensitivity": format_metric(kcv_sens),
       "Mean specificity": format_metric(kcv_spec),
       "Mean precision": format_metric(kcv_prec),
       "Mean F1 score": format_metric(kcv_f1)
       }
    
    # find the best model hyperparameter(s) based on the mode 
    final_params = pd.Series([tuple(sorted(p.items())) for p in selected_params]).mode()[0]
    final_params = dict(final_params)
    
    # build final model
    kcv_model = classifier.set_params(**final_params)
    # save k-fold model
    joblib.dump(kcv_model, model_path)

    print(kcv_model)
    print("K-fold cross-validation performance metrics:")
    for name, val in kcv_metrics.items():
        print(f"  {name}: {val}")




### Leave-one-out cross-validation (LOOCV) function 


def loocv_val(X, Y, groups, classifier, param_grid,
              model_path, r_state=0):
    
    """
    Applies nested leave-one-out cross validation (LOOCV) for a classifier.

    Parameters:
    - X: features (pandas dataframe)
    - Y: class labels (series)
    - groups: group labels (for paired splitting)
    - classifier: machine learning classifier 
    - param_grid: dict or list of dicts for classifier hyperparameter tuning using GridSearchCV
    - model_path: full path to save the final model (.pkl)
    - r_state: random seed

    Returns:
    - best_model: classifier model with tuned hyperparameters
    - metrics: dictionary of performance metrics for LOOCV cross validation
    """

    # define cv split (nested group cv)
    outer_cv = LeaveOneGroupOut()
    inner_cv = GroupKFold(n_splits=5, shuffle=True, random_state=r_state)

    # storing results
    outer_results = []
    selected_params = []

    # for each 'k' fold:
    for fold, (train_index, test_index) in enumerate(outer_cv.split(X, Y, groups), 1):
       
       print(f"Outer fold {fold}/{outer_cv.get_n_splits(groups=groups)}", flush=True)
       
       # split into the training and testing folds 
       X_train, X_test = X.iloc[train_index], X.iloc[test_index]
       Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
       groups_train = groups[train_index]

       # hyperparameter tuning for classifier using param_grid
       grid_search = GridSearchCV(
           estimator=classifier,
           param_grid=param_grid,
           cv=inner_cv,
           scoring='roc_auc',
           n_jobs=-1) 
       
       grid_search.fit(X_train, Y_train, groups=groups_train)
       
       # selected parameters
       selected_params.append(grid_search.best_params_)
       # best model
       best_model = grid_search.best_estimator_
       
       # test best model on the outer test fold
       Y_pred = best_model.predict(X_test)
       Y_prob = best_model.predict_proba(X_test)[:,1]
       
       # calculate performance metrics
       auc = roc_auc_score(Y_test, Y_prob)
       acc = accuracy_score(Y_test, Y_pred)
       tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
       sens = tp / (tp + fn)
       spec = tn / (tn + fp)
       # for precision and f1 - account for divison by zeros 
       if (tp + fp) == 0:   
          prec = 0.0
       else:
          prec = tp / (tp + fp)
       if (prec + sens) == 0:
          f1 = 0.0
       else:
          f1 = 2 * (prec * sens) / (prec + sens)

       outer_results.append({
          'auc': auc,
          'acc': acc,
          'sens': sens,
          'spec': spec,
          'prec': prec,
          'f1': f1
          })
    
    # extract results 
    loocv_auc  = [r['auc']  for r in outer_results]
    loocv_acc  = [r['acc']  for r in outer_results]
    loocv_sens = [r['sens'] for r in outer_results]
    loocv_spec = [r['spec'] for r in outer_results]
    loocv_prec = [r['prec'] for r in outer_results]
    loocv_f1 = [r['f1'] for r in outer_results]
    
    # function to format mean ± std + 95% CI
    def format_metric(values):
       mean = np.mean(values)
       std = np.std(values)
       ci_lower = np.percentile(values, 2.5)
       ci_upper = np.percentile(values, 97.5)
       return f"{mean:.3f} ± {std:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])"
    
    loocv_metrics = {
       "Mean AUC": format_metric(loocv_auc),
       "Mean accuracy": format_metric(loocv_acc),
       "Mean sensitivity": format_metric(loocv_sens),
       "Mean specificity": format_metric(loocv_spec),
       "Mean precision": format_metric(loocv_prec),
       "Mean F1 score": format_metric(loocv_f1)
       }
    
    # find the best model hyperparameter(s) based on the mode 
    final_params = pd.Series([tuple(sorted(p.items())) for p in selected_params]).mode()[0]
    final_params = dict(final_params)
    
    # build final model
    loocv_model = classifier.set_params(**final_params)
    # save LOOCV model
    joblib.dump(loocv_model, model_path)

    print(loocv_model)
    print("LOOCV cross-validation performance metrics:")
    for name, val in loocv_metrics.items():
        print(f"  {name}: {val}")
    
