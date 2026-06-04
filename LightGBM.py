# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 01:36:34 2026

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
"""

import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import multiprocessing
import matplotlib

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties, fontManager

# =========================

# =========================
candidate_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'Noto Sans CJK SC']
available_fonts = {f.name for f in fontManager.ttflist}

chosen_font = None
for f in candidate_fonts:
    if f in available_fonts:
        chosen_font = f
        break

if chosen_font is None:
    print("Warning: No common Chinese font was found. Chinese characters in the figure may not be displayed properly.")
    chosen_font = 'DejaVu Sans'

print(f"Current font used for Chinese display: {chosen_font}")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [chosen_font]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'   
plt.rcParams['figure.dpi'] = 300

font_cn = FontProperties(family=chosen_font)
font_en = FontProperties(family='Times New Roman')

# =========================
def dic2txt(params, txt_path):
    """ txt"""
    with open(txt_path, 'w', encoding='utf-8') as f:
        for k, v in params.items():
            f.write(f"{k}:{v}\n")


def txt2dic(txtname):
    para_dic = {}
    with open(txtname, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            para_dic[k] = v
    return para_dic


def cv_results_to_txt(txtname, gsearch):
    with open(txtname, 'w', encoding='utf-8') as f:
        rank_test_score = str(gsearch.cv_results_['rank_test_score'])
        std_test_score = str(gsearch.cv_results_['std_test_score'])
        mean_test_score = str(gsearch.cv_results_['mean_test_score'])
        params = gsearch.cv_results_['params']

        for i in range(len(params)):
            s = str(params[i]).replace('[', '').replace(']', '')
            s = s.replace("'", '').replace(',', '') + '\n'
            f.write(s)

        f.write(rank_test_score + '\n')
        f.write(mean_test_score + '\n')
        f.write(std_test_score + '\n')


def evaluate_regression(y_true, y_pred, name='Dataset'):
    r2 = metrics.r2_score(y_true, y_pred)
    mae = MAE(y_true, y_pred)
    mse = MSE(y_true, y_pred)
    rmse = mse ** 0.5

    print(f'{name} R2:   {r2:.6f}')
    print(f'{name} MAE:  {mae:.6f}')
    print(f'{name} MSE:  {mse:.6f}')
    print(f'{name} RMSE: {rmse:.6f}')
    print('-' * 40)

    return r2, mae, mse, rmse


def zero_formatter(x, pos):
    if abs(x) < 1e-8:
        return '0'
    return f'{x:.1f}'


def is_between_lines(x, y):
    return (y >= 0.8 * x) and (y <= 1.2 * x)


def run_grid_search_step(best_params, cv_params, X_train, y_train, kf, n_jobs_search, txtfilename):
    """
     GridSearchCV
    """
    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        metric='rmse',
        learning_rate=best_params['learning_rate'],
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        num_leaves=best_params['num_leaves'],
        max_bin=best_params['max_bin'],
        min_child_samples=best_params['min_child_samples'],
        colsample_bytree=best_params['colsample_bytree'],
        subsample=best_params['subsample'],
        subsample_freq=best_params['subsample_freq'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        min_split_gain=best_params['min_split_gain'],
        random_state=42,
        n_jobs=1,                
        verbosity=-1,
        force_col_wise=True
    )

    gsearch = GridSearchCV(
        estimator=model_lgb,
        param_grid=cv_params,
        scoring='neg_mean_squared_error',
        cv=kf,
        verbose=1,
        n_jobs=n_jobs_search,   # Outer-level parallelization
        pre_dispatch='2*n_jobs'
    )

    gsearch.fit(X_train, y_train)
    cv_results_to_txt(txtfilename, gsearch)
    return gsearch.best_params_


# =========================
# main
# =========================
if __name__ == '__main__':

    # -------------------------
    # 1. CPU set
    # -------------------------
    total_cpu = multiprocessing.cpu_count()

    n_jobs_search = max(1, min(8, total_cpu // 2))
    n_jobs_train = max(1, min(8, total_cpu // 2))

    print("Total CPU cores:", total_cpu)
    print("GridSearchCV n_jobs:", n_jobs_search)
    print("Final model n_jobs:", n_jobs_train)

    # # Five-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------
    # 2. Parameter settings
    # -------------------------
    data_path = r' "data/"'
    TxtFilePath = r'lightgbm_parameters_normal_regression_step_search.txt'
    model_path = 'lgb_model_normal_regression_step_search.pkl'

    outputCol = 'Kt'
    inputCol = ['RcHm0', 'BL', 'DAHm0', 'Sop', 'hchd', 'Hm0Hd', 'DCHm0']

    # -------------------------
    # 3. data
    # -------------------------
    data = pd.read_csv(data_path, encoding='gbk')
    data = data[inputCol + [outputCol]].dropna().reset_index(drop=True)

    X_all = data[inputCol].values
    y_all = data[outputCol].values.ravel()

    print("Total samples:", len(X_all))
    print("Feature dimension:", X_all.shape[1])

    # -------------------------
    # 4. Split into training set / validation set / test set
    # -------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,   # 0.25 * 0.8 = 0.2
        random_state=42,
        shuffle=True
    )

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    # -------------------------
    # # Initialize parameter file
    # -------------------------
    if os.path.isfile(TxtFilePath):
        First_Search_Flag = 1
    else:
        First_Search_Flag = 0

    best_params = {}

    if First_Search_Flag == 0:
        best_params = {
            'learning_rate_coarse': 0.05,
            'learning_rate_fine': 0.03,
            'num_leaves': 10,
            'n_estimators_coarse': 800,
            'n_estimators_fine': 800,
            'max_bin': 255,
            'min_child_samples': 30,
            'max_depth': 4,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'subsample_freq': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'min_split_gain': 0.05,
            'step0': 0,
            'step1': 0,
            'step2': 0,
            'step3': 0,
            'step4': 0,
            'step5': 0
        }
        dic2txt(best_params, TxtFilePath)
        print("Initial parameter file has been written.")
    else:
        para_dic = txt2dic(TxtFilePath)
        best_params = para_dic.copy()

        best_params['learning_rate_coarse'] = float(para_dic['learning_rate_coarse'])
        best_params['learning_rate_fine'] = float(para_dic['learning_rate_fine'])
        best_params['n_estimators_coarse'] = int(para_dic['n_estimators_coarse'])
        best_params['n_estimators_fine'] = int(para_dic['n_estimators_fine'])
        best_params['num_leaves'] = int(para_dic['num_leaves'])
        best_params['max_depth'] = int(para_dic['max_depth'])
        best_params['max_bin'] = int(para_dic['max_bin'])
        best_params['min_child_samples'] = int(para_dic['min_child_samples'])
        best_params['colsample_bytree'] = float(para_dic['colsample_bytree'])
        best_params['subsample'] = float(para_dic['subsample'])
        best_params['subsample_freq'] = int(para_dic['subsample_freq'])
        best_params['reg_alpha'] = float(para_dic['reg_alpha'])
        best_params['reg_lambda'] = float(para_dic['reg_lambda'])
        best_params['min_split_gain'] = float(para_dic['min_split_gain'])
        best_params['step0'] = int(para_dic['step0'])
        best_params['step1'] = int(para_dic['step1'])
        best_params['step2'] = int(para_dic['step2'])
        best_params['step3'] = int(para_dic['step3'])
        best_params['step4'] = int(para_dic['step4'])
        best_params['step5'] = int(para_dic['step5'])

        print("已读取已有参数")

    # -------------------------
    # 6. Step0: find n_estimators
    # -------------------------
    if best_params['step0'] == 0:
        print("Step0: coarse search for n_estimators")

        other_params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'learning_rate': best_params['learning_rate_coarse'],
            'n_estimators': best_params['n_estimators_coarse'],
            'max_depth': best_params['max_depth'],
            'num_leaves': best_params['num_leaves'],
            'max_bin': best_params['max_bin'],
            'min_child_samples': best_params['min_child_samples'],
            'colsample_bytree': best_params['colsample_bytree'],
            'subsample': best_params['subsample'],
            'subsample_freq': best_params['subsample_freq'],
            'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'],
            'min_split_gain': best_params['min_split_gain'],
            'verbosity': -1,
            'seed': 42,
            'force_col_wise': True
        }

        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

        cv_results = lgb.cv(
            params=other_params,
            train_set=lgb_train,
            folds=kf,
            stratified=False,
            shuffle=True,
            seed=42
        )

        rmse_mean = cv_results['rmse-mean']
        optimal_n_estimators = np.argmin(rmse_mean) + 1

        pd.DataFrame(rmse_mean).to_csv(
            'rmse_mean_' + str(other_params['learning_rate']).replace('.', '_') + '.csv',
            index=True
        )

        best_params['n_estimators_coarse'] = optimal_n_estimators
        best_params['step0'] = 1
        dic2txt(best_params, TxtFilePath)

        print("Step0 完成，最佳 coarse n_estimators:", optimal_n_estimators)

    # -------------------------
    # 7. Step1:  max_depth and num_leaves
    # -------------------------
    if best_params['step1'] == 0:
        print("Step1: adjusting num_leaves and max_depth")

        best_params['learning_rate'] = best_params['learning_rate_coarse']
        best_params['n_estimators'] = best_params['n_estimators_coarse']

        cv_params = {
            'max_depth': [3, 4, 5, 6],
            'num_leaves': [7, 10, 15, 20]
        }

        best = run_grid_search_step(
            best_params=best_params,
            cv_params=cv_params,
            X_train=X_train,
            y_train=y_train,
            kf=kf,
            n_jobs_search=n_jobs_search,
            txtfilename='gsearch_step1.txt'
        )

        best_params['num_leaves'] = best['num_leaves']
        best_params['max_depth'] = best['max_depth']
        best_params['step1'] = 1
        dic2txt(best_params, TxtFilePath)

        print('best num_leaves is', best_params['num_leaves'])
        print('best max_depth is', best_params['max_depth'])

    # -------------------------
    # 8. Step2:  max_bin and min_child_samples
    # -------------------------
    if best_params['step2'] == 0:
        print("Step2: adjusting max_bin and min_child_samples")

        best_params['learning_rate'] = best_params['learning_rate_coarse']
        best_params['n_estimators'] = best_params['n_estimators_coarse']

        cv_params = {
            'max_bin': [127, 191, 255],
            'min_child_samples': [20, 30, 40, 50]
        }

        best = run_grid_search_step(
            best_params=best_params,
            cv_params=cv_params,
            X_train=X_train,
            y_train=y_train,
            kf=kf,
            n_jobs_search=n_jobs_search,
            txtfilename='gsearch_step2.txt'
        )

        best_params['max_bin'] = best['max_bin']
        best_params['min_child_samples'] = best['min_child_samples']
        best_params['step2'] = 1
        dic2txt(best_params, TxtFilePath)

        print('best min_child_samples is', best_params['min_child_samples'])
        print('best max_bin is', best_params['max_bin'])

    # -------------------------
    # 9. Step3:  colsample_bytree / subsample / subsample_freq
    # -------------------------
    if best_params['step3'] == 0:
        print("Step3: adjusting colsample_bytree, subsample, and subsample_freq")

        best_params['learning_rate'] = best_params['learning_rate_coarse']
        best_params['n_estimators'] = best_params['n_estimators_coarse']

        cv_params = {
            'colsample_bytree': [0.7, 0.8, 0.9],
            'subsample': [0.7, 0.8, 0.9],
            'subsample_freq': [1, 3, 5, 7]
        }

        best = run_grid_search_step(
            best_params=best_params,
            cv_params=cv_params,
            X_train=X_train,
            y_train=y_train,
            kf=kf,
            n_jobs_search=n_jobs_search,
            txtfilename='gsearch_step3.txt'
        )

        best_params['colsample_bytree'] = best['colsample_bytree']
        best_params['subsample'] = best['subsample']
        best_params['subsample_freq'] = best['subsample_freq']
        best_params['step3'] = 1
        dic2txt(best_params, TxtFilePath)

        print('best colsample_bytree is:', best_params['colsample_bytree'])
        print('best subsample is:', best_params['subsample'])
        print('best subsample_freq is', best_params['subsample_freq'])

    # -------------------------
    # 10. Step4: Tune the regularization term
    # -------------------------
    if best_params['step4'] == 0:
        print("Step4: adjusting reg_alpha, reg_lambda, and min_split_gain")

        best_params['learning_rate'] = best_params['learning_rate_coarse']
        best_params['n_estimators'] = best_params['n_estimators_coarse']

        cv_params = {
            'reg_alpha': [0.0, 0.2, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0, 3.0],
            'min_split_gain': [0.0, 0.02, 0.05, 0.1]
        }

        best = run_grid_search_step(
            best_params=best_params,
            cv_params=cv_params,
            X_train=X_train,
            y_train=y_train,
            kf=kf,
            n_jobs_search=n_jobs_search,
            txtfilename='gsearch_step4.txt'
        )

        best_params['reg_alpha'] = best['reg_alpha']
        best_params['reg_lambda'] = best['reg_lambda']
        best_params['min_split_gain'] = best['min_split_gain']
        best_params['step4'] = 1
        dic2txt(best_params, TxtFilePath)

        print('best reg_alpha is:', best_params['reg_alpha'])
        print('best reg_lambda is:', best_params['reg_lambda'])
        print('best min_split_gain is:', best_params['min_split_gain'])

    # -------------------------
    # 11. Step5:  Use a smaller learning_rate and then search for n_estimators
    # -------------------------
    if best_params['step5'] == 0:
        print("Step5: fine search for n_estimators with smaller learning_rate")

        other_params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'learning_rate': best_params['learning_rate_fine'],
            'n_estimators': best_params['n_estimators_fine'],
            'max_depth': best_params['max_depth'],
            'num_leaves': best_params['num_leaves'],
            'max_bin': best_params['max_bin'],
            'min_child_samples': best_params['min_child_samples'],
            'colsample_bytree': best_params['colsample_bytree'],
            'subsample': best_params['subsample'],
            'subsample_freq': best_params['subsample_freq'],
            'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'],
            'min_split_gain': best_params['min_split_gain'],
            'verbosity': -1,
            'seed': 42,
            'force_col_wise': True
        }

        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

        cv_results = lgb.cv(
            params=other_params,
            train_set=lgb_train,
            folds=kf,
            stratified=False,
            shuffle=True,
            seed=42
        )

        rmse_mean_fine = cv_results['rmse-mean']
        optimal_n_estimators_fine = np.argmin(rmse_mean_fine) + 1

        pd.DataFrame(rmse_mean_fine).to_csv(
            'rmse_mean_fine_' + str(other_params['learning_rate']).replace('.', '_') + '.csv',
            index=True
        )

        best_params['n_estimators_fine'] = optimal_n_estimators_fine
        best_params['step5'] = 1
        dic2txt(best_params, TxtFilePath)

        print("Step5 完成，最佳 fine n_estimators:", optimal_n_estimators_fine)

    # -------------------------
    # 12.  Output the final parameters
    # -------------------------
    final_model_params = {
        'learning_rate': best_params['learning_rate_fine'],
        'n_estimators': best_params['n_estimators_fine'],
        'max_depth': best_params['max_depth'],
        'num_leaves': best_params['num_leaves'],
        'max_bin': best_params['max_bin'],
        'min_child_samples': best_params['min_child_samples'],
        'colsample_bytree': best_params['colsample_bytree'],
        'subsample': best_params['subsample'],
        'subsample_freq': best_params['subsample_freq'],
        'reg_alpha': best_params['reg_alpha'],
        'reg_lambda': best_params['reg_lambda'],
        'min_split_gain': best_params['min_split_gain']
    }

    print(" Final optimal parameters：")
    for k, v in final_model_params.items():
        print(f"{k}: {v}")

    # -------------------------
    # 13.  Final model training
    # -------------------------
    model = lgb.LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        metric='rmse',
        random_state=42,
        n_jobs=n_jobs_train,
        verbosity=-1,
        force_col_wise=True,
        **final_model_params
    )

    retrain_flag = True

    if (not retrain_flag) and os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Existing model has been loaded.：", model_path)
    else:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='l2',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(50)
            ],
            feature_name=inputCol
        )
        joblib.dump(model, model_path)
     
        print(f"Model has been saved to: {model_path}")

    # -------------------------
    #
    # -------------------------
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    r2_train, mae_train, mse_train, rmse_train = evaluate_regression(y_train, pred_train, 'Train')
    r2_val, mae_val, mse_val, rmse_val = evaluate_regression(y_val, pred_val, 'Validation')
    r2_test, mae_test, mse_test, rmse_test = evaluate_regression(y_test, pred_test, 'Test')

    # -------------------------
   # Overfitting risk assessment
    # -------------------------
    r2_gap_val = r2_train - r2_val
    r2_gap_test = r2_train - r2_test
    rmse_ratio_val = rmse_val / rmse_train if rmse_train > 1e-12 else np.inf
    rmse_ratio_test = rmse_test / rmse_train if rmse_train > 1e-12 else np.inf

    print(f"Train-Val R2 gap: {r2_gap_val:.4f}")
    print(f"Train-Test R2 gap: {r2_gap_test:.4f}")
    print(f"Val/Train RMSE ratio: {rmse_ratio_val:.4f}")
    print(f"Test/Train RMSE ratio: {rmse_ratio_test:.4f}")

    val_overfit_strong = (r2_gap_val > 0.10 and rmse_ratio_val > 1.30)
    test_overfit_strong = (r2_gap_test > 0.10 and rmse_ratio_test > 1.30)

    val_overfit_mild = (r2_gap_val > 0.05 and rmse_ratio_val > 1.15)
    test_overfit_mild = (r2_gap_test > 0.05 and rmse_ratio_test > 1.15)

    if val_overfit_strong and test_overfit_strong:
        print("Warning: The model shows obvious overfitting on both the validation set and the test set.")
    elif test_overfit_strong:
        print("Warning: The model shows obvious overfitting on the test set.")
    elif val_overfit_strong:
        print("Warning: The model shows obvious overfitting on the validation set.")
    elif val_overfit_mild and test_overfit_mild:
        print("Note: The model shows a certain tendency of overfitting on both the validation set and the test set.")
    elif test_overfit_mild:
        print("Note: The model shows a certain tendency of overfitting on the test set.")
    elif val_overfit_mild:
        print("Note: The model shows a certain tendency of overfitting on the validation set.")
    else:
        print("The model shows normal generalization performance, with no obvious overfitting observed.")

    if r2_train > 0.98 and (val_overfit_mild or test_overfit_mild):
        print(
            "Additional note: The training set fitting performance is very high, "
            "so further attention should be paid to overfitting."
        )

    if val_overfit_strong or test_overfit_strong:
        overfit_summary = "Obvious overfitting"
    elif val_overfit_mild or test_overfit_mild:
        overfit_summary = "Mild overfitting"
    else:
        overfit_summary = "No obvious overfitting"

    print(f"Overfitting assessment: {overfit_summary}")
    

    # -------------------------
    plt.figure(figsize=(8, 8), dpi=300)

    plt.plot(y_train, pred_train, 'ko',
             markersize=3,
             label='Train set')

    plt.plot(y_test, pred_test, 'ro',
             markersize=4,
             markerfacecolor='none',
             markeredgewidth=1.2,
             label='Test set')

    plt.tick_params(axis='both', direction='in', length=8, width=2)
    

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.xlabel(r'$\mathit{K}_{\mathrm{t,obs}}[-]$', fontsize=30, fontname='Times New Roman')
    plt.ylabel(r'$\mathit{K}_{\mathrm{t,pred}}[-]$', fontsize=30, fontname='Times New Roman')

    plt.text(
        0.05, 0.96,
    
        f'R$^2$: {r2_test:.4f}\nRMSE: {rmse_test:.4f}',
        transform=plt.gca().transAxes,
        fontsize=22,
        verticalalignment='top',
        horizontalalignment='left',
        fontname='Times New Roman'
    )

    plt.tick_params(labelsize=26)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
      label.set_fontname('Times New Roman')
    plt.plot([0, 1.0], [0, 1.0], '--', color='gray')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    ax.xaxis.set_major_formatter(FuncFormatter(zero_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(zero_formatter))
    #font_cn_legend = FontProperties(family=chosen_font, size=22)
    font_en_legend = FontProperties(
    family='Times New Roman',
    size=22
)
    legend = plt.legend(
      loc='upper right',
      bbox_to_anchor=(1.0, 0.17),
      frameon=False,
      prop=font_en_legend,
      markerscale=2.4,      
      handletextpad=0.3,    
      labelspacing=0.3,     
      borderpad=0.2         
    )

    plt.tight_layout()
    plt.savefig('train_test_scatter_fast_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

 

 # -------------------------
    booster = model.booster_
    importance = booster.feature_importance(importance_type='split')
    total_splits = np.sum(importance)

    if total_splits == 0:
         importance_ratio = np.zeros_like(importance, dtype=float)
    else:
         importance_ratio = (importance / total_splits) * 100

    feature_names = inputCol
    sorted_idx = np.argsort(importance_ratio)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importance_ratio = importance_ratio[sorted_idx]

    plt.figure(figsize=(12.5, 8), dpi=600)
    ax = plt.gca()

    plt.tick_params(
       axis='both',
       direction='in',
       length=8,
       width=2
    )

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.barh(
        range(len(sorted_importance_ratio)),
        sorted_importance_ratio,
        align='center'
    )

    plt.xlim(
        0,
        max(sorted_importance_ratio) * 1.1
        if max(sorted_importance_ratio) > 0 else 1
    )

    plt.xticks(
        fontsize=28,
        fontname='Times New Roman'
    )

    plt.yticks(
        range(len(sorted_importance_ratio)),
        sorted_names,
        fontsize=20,
        fontname='Times New Roman'
    )

    plt.xlabel(
        'Contribution Degree(%)',
        fontsize=26,
        fontname='Times New Roman'
    )

    plt.title(
        'Feature Importance (FI)',
        fontsize=26,
        fontname='Times New Roman'
    )

    for i, v in enumerate(sorted_importance_ratio):
        plt.text(
            v + 0.2,
            i,
            f"{v:.1f}%",
            fontsize=24,
            fontname='Times New Roman'
     )

    plt.tight_layout()

    plt.savefig(
        'feature_importance_fast_grid.png',
        dpi=600,
        bbox_inches='tight'
     )

    plt.show()

    # -------------------------

    # -------------------------
  
    feature_importance_df = pd.DataFrame({
      'feature_name': inputCol,
      'importance_split': importance,
      'importance_ratio_percent': importance_ratio
  }).sort_values(by='importance_ratio_percent', ascending=False)

    feature_importance_df.to_csv('feature_importance_fast_grid.csv', index=False, encoding='utf-8-sig')
    print("Feature importance has been saved：feature_importance_fast_grid.csv")

    print("Program execution completed.")

    # -------------------------
    # 1. CPU set
    # -------------------------
    total_cpu = multiprocessing.cpu_count()

    n_jobs_search = max(1, min(8, total_cpu // 2))
    n_jobs_train = max(1, min(8, total_cpu // 2))

    print("Total CPU cores:", total_cpu)
    print("GridSearchCV n_jobs:", n_jobs_search)
    print("Final model n_jobs:", n_jobs_train)

   # Five-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------
    # 2. 参数设置
    # -------------------------
    data_path = r' "data/"'
    TxtFilePath = r'lightgbm_parameters_normal_regression_step_search.txt'
    model_path = 'lgb_model_normal_regression_step_search.pkl'

    outputCol = 'Kt'
    inputCol = ['RcHm0', 'BL', 'DAHm0', 'Sop', 'hchd', 'Hm0Hd', 'DCHm0']

    # -------------------------
    # 3. 读取数据
    # -------------------------
    data = pd.read_csv(data_path, encoding='gbk')
    data = data[inputCol + [outputCol]].dropna().reset_index(drop=True)

    X_all = data[inputCol].values
    y_all = data[outputCol].values.ravel()

    print("Total samples:", len(X_all))
    print("Feature dimension:", X_all.shape[1])

    # -------------------------
    # 4. 划分训练集 / 验证集 / 测试集
    # -------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,   # 0.25 * 0.8 = 0.2
        random_state=42,
        shuffle=True
    )

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    # -------------------------
    # 5. 初始化参数文件
    # -------------------------
    if os.path.isfile(TxtFilePath):
        First_Search_Flag = 1
    else:
        First_Search_Flag = 0

    best_params = {}

    if First_Search_Flag == 0:
        best_params = {
            'learning_rate_coarse': 0.05,
            'learning_rate_fine': 0.03,
            'num_leaves': 10,
            'n_estimators_coarse': 800,
            'n_estimators_fine': 800,
            'max_bin': 255,
            'min_child_samples': 30,
            'max_depth': 4,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'subsample_freq': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'min_split_gain': 0.05,
            'step0': 0,
            'step1': 0,
            'step2': 0,
            'step3': 0,
            'step4': 0,
            'step5': 0
        }
        dic2txt(best_params, TxtFilePath)
        print("已写入初始参数文件")
    else:
        para_dic = txt2dic(TxtFilePath)
        best_params = para_dic.copy()

        best_params['learning_rate_coarse'] = float(para_dic['learning_rate_coarse'])
        best_params['learning_rate_fine'] = float(para_dic['learning_rate_fine'])
        best_params['n_estimators_coarse'] = int(para_dic['n_estimators_coarse'])
        best_params['n_estimators_fine'] = int(para_dic['n_estimators_fine'])
        best_params['num_leaves'] = int(para_dic['num_leaves'])
        best_params['max_depth'] = int(para_dic['max_depth'])
        best_params['max_bin'] = int(para_dic['max_bin'])
        best_params['min_child_samples'] = int(para_dic['min_child_samples'])
        best_params['colsample_bytree'] = float(para_dic['colsample_bytree'])
        best_params['subsample'] = float(para_dic['subsample'])
        best_params['subsample_freq'] = int(para_dic['subsample_freq'])
        best_params['reg_alpha'] = float(para_dic['reg_alpha'])
        best_params['reg_lambda'] = float(para_dic['reg_lambda'])
        best_params['min_split_gain'] = float(para_dic['min_split_gain'])
        best_params['step0'] = int(para_dic['step0'])
        best_params['step1'] = int(para_dic['step1'])
        best_params['step2'] = int(para_dic['step2'])
        best_params['step3'] = int(para_dic['step3'])
        best_params['step4'] = int(para_dic['step4'])
        best_params['step5'] = int(para_dic['step5'])

        print("已读取已有参数")

    # -------------------------
    # 6. Step0: 粗略找 n_estimators
    # -------------------------
    if best_params['step0'] == 0:
        print("Step0: coarse search for n_estimators")

        other_params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'learning_rate': best_params['learning_rate_coarse'],
            'n_estimators': best_params['n_estimators_coarse'],
            'max_depth': best_params['max_depth'],
            'num_leaves': best_params['num_leaves'],
            'max_bin': best_params['max_bin'],
            'min_child_samples': best_params['min_child_samples'],
            'colsample_bytree': best_params['colsample_bytree'],
            'subsample': best_params['subsample'],
            'subsample_freq': best_params['subsample_freq'],
            'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'],
            'min_split_gain': best_params['min_split_gain'],
            'verbosity': -1,
            'seed': 42,
            'force_col_wise': True
        }

        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

        cv_results = lgb.cv(
            params=other_params,
            train_set=lgb_train,
            folds=kf,
            stratified=False,
            shuffle=True,
            seed=42
        )

        rmse_mean = cv_results['rmse-mean']
        optimal_n_estimators = np.argmin(rmse_mean) + 1

        pd.DataFrame(rmse_mean).to_csv(
            'rmse_mean_' + str(other_params['learning_rate']).replace('.', '_') + '.csv',
            index=True
        )

        best_params['n_estimators_coarse'] = optimal_n_estimators
        best_params['step0'] = 1
        dic2txt(best_params, TxtFilePath)

        print("Step0 完成，最佳 coarse n_estimators:", optimal_n_estimators)

    # -------------------------
    # 7. Step1: 调 max_depth 和 num_leaves
    # -------------------------
    if best_params['step1'] == 0:
        print("Step1: adjusting num_leaves and max_depth")

        best_params['learning_rate'] = best_params['learning_rate_coarse']
        best_params['n_estimators'] = best_params['n_estimators_coarse']

        cv_params = {
            'max_depth': [3, 4, 5, 6],
            'num_leaves': [7, 10, 15, 20]
        }

        best = run_grid_search_step(
            best_params=best_params,
            cv_params=cv_params,
            X_train=X_train,
            y_train=y_train,
            kf=kf,
            n_jobs_search=n_jobs_search,
            txtfilename='gsearch_step1.txt'
        )

        best_params['num_leaves'] = best['num_leaves']
        best_params['max_depth'] = best['max_depth']
        best_params['step1'] = 1
        dic2txt(best_params, TxtFilePath)

        print('best num_leaves is', best_params['num_leaves'])
        print('best max_depth is', best_params['max_depth'])

    # -------------------------
    # 8. Step2: 调 max_bin 和 min_child_samples
    # -------------------------
    if best_params['step2'] == 0:
        print("Step2: adjusting max_bin and min_child_samples")

        best_params['learning_rate'] = best_params['learning_rate_coarse']
        best_params['n_estimators'] = best_params['n_estimators_coarse']

        cv_params = {
            'max_bin': [127, 191, 255],
            'min_child_samples': [20, 30, 40, 50]
        }

        best = run_grid_search_step(
            best_params=best_params,
            cv_params=cv_params,
            X_train=X_train,
            y_train=y_train,
            kf=kf,
            n_jobs_search=n_jobs_search,
            txtfilename='gsearch_step2.txt'
        )

        best_params['max_bin'] = best['max_bin']
        best_params['min_child_samples'] = best['min_child_samples']
        best_params['step2'] = 1
        dic2txt(best_params, TxtFilePath)

        print('best min_child_samples is', best_params['min_child_samples'])
        print('best max_bin is', best_params['max_bin'])

    # -------------------------
    # 9. Step3: 调 colsample_bytree / subsample / subsample_freq
    # -------------------------
    if best_params['step3'] == 0:
        print("Step3: adjusting colsample_bytree, subsample, and subsample_freq")

        best_params['learning_rate'] = best_params['learning_rate_coarse']
        best_params['n_estimators'] = best_params['n_estimators_coarse']

        cv_params = {
            'colsample_bytree': [0.7, 0.8, 0.9],
            'subsample': [0.7, 0.8, 0.9],
            'subsample_freq': [1, 3, 5, 7]
        }

        best = run_grid_search_step(
            best_params=best_params,
            cv_params=cv_params,
            X_train=X_train,
            y_train=y_train,
            kf=kf,
            n_jobs_search=n_jobs_search,
            txtfilename='gsearch_step3.txt'
        )

        best_params['colsample_bytree'] = best['colsample_bytree']
        best_params['subsample'] = best['subsample']
        best_params['subsample_freq'] = best['subsample_freq']
        best_params['step3'] = 1
        dic2txt(best_params, TxtFilePath)

        print('best colsample_bytree is:', best_params['colsample_bytree'])
        print('best subsample is:', best_params['subsample'])
        print('best subsample_freq is', best_params['subsample_freq'])

    # -------------------------
    # 10. Step4: 调正则项
    # -------------------------
    if best_params['step4'] == 0:
        print("Step4: adjusting reg_alpha, reg_lambda, and min_split_gain")

        best_params['learning_rate'] = best_params['learning_rate_coarse']
        best_params['n_estimators'] = best_params['n_estimators_coarse']

        cv_params = {
            'reg_alpha': [0.0, 0.2, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0, 3.0],
            'min_split_gain': [0.0, 0.02, 0.05, 0.1]
        }

        best = run_grid_search_step(
            best_params=best_params,
            cv_params=cv_params,
            X_train=X_train,
            y_train=y_train,
            kf=kf,
            n_jobs_search=n_jobs_search,
            txtfilename='gsearch_step4.txt'
        )

        best_params['reg_alpha'] = best['reg_alpha']
        best_params['reg_lambda'] = best['reg_lambda']
        best_params['min_split_gain'] = best['min_split_gain']
        best_params['step4'] = 1
        dic2txt(best_params, TxtFilePath)

        print('best reg_alpha is:', best_params['reg_alpha'])
        print('best reg_lambda is:', best_params['reg_lambda'])
        print('best min_split_gain is:', best_params['min_split_gain'])

    # -------------------------
    # 11. Step5: 用更小 learning_rate 再找 n_estimators
    # -------------------------
    if best_params['step5'] == 0:
        print("Step5: fine search for n_estimators with smaller learning_rate")

        other_params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'learning_rate': best_params['learning_rate_fine'],
            'n_estimators': best_params['n_estimators_fine'],
            'max_depth': best_params['max_depth'],
            'num_leaves': best_params['num_leaves'],
            'max_bin': best_params['max_bin'],
            'min_child_samples': best_params['min_child_samples'],
            'colsample_bytree': best_params['colsample_bytree'],
            'subsample': best_params['subsample'],
            'subsample_freq': best_params['subsample_freq'],
            'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'],
            'min_split_gain': best_params['min_split_gain'],
            'verbosity': -1,
            'seed': 42,
            'force_col_wise': True
        }

        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

        cv_results = lgb.cv(
            params=other_params,
            train_set=lgb_train,
            folds=kf,
            stratified=False,
            shuffle=True,
            seed=42
        )

        rmse_mean_fine = cv_results['rmse-mean']
        optimal_n_estimators_fine = np.argmin(rmse_mean_fine) + 1

        pd.DataFrame(rmse_mean_fine).to_csv(
            'rmse_mean_fine_' + str(other_params['learning_rate']).replace('.', '_') + '.csv',
            index=True
        )

        best_params['n_estimators_fine'] = optimal_n_estimators_fine
        best_params['step5'] = 1
        dic2txt(best_params, TxtFilePath)

        print("Step5 完成，最佳 fine n_estimators:", optimal_n_estimators_fine)

    # -------------------------
    # 12. 输出最终参数
    # -------------------------
    final_model_params = {
        'learning_rate': best_params['learning_rate_fine'],
        'n_estimators': best_params['n_estimators_fine'],
        'max_depth': best_params['max_depth'],
        'num_leaves': best_params['num_leaves'],
        'max_bin': best_params['max_bin'],
        'min_child_samples': best_params['min_child_samples'],
        'colsample_bytree': best_params['colsample_bytree'],
        'subsample': best_params['subsample'],
        'subsample_freq': best_params['subsample_freq'],
        'reg_alpha': best_params['reg_alpha'],
        'reg_lambda': best_params['reg_lambda'],
        'min_split_gain': best_params['min_split_gain']
    }

    print("最终最优参数：")
    for k, v in final_model_params.items():
        print(f"{k}: {v}")

    # -------------------------
    # 13. 最终模型训练
    # -------------------------
    model = lgb.LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        metric='rmse',
        random_state=42,
        n_jobs=n_jobs_train,
        verbosity=-1,
        force_col_wise=True,
        **final_model_params
    )

    retrain_flag = True

    if (not retrain_flag) and os.path.exists(model_path):
        model = joblib.load(model_path)
        print("已加载已有模型：", model_path)
    else:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='l2',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(50)
            ],
            feature_name=inputCol
        )
        joblib.dump(model, model_path)
        print("模型已保存到：", model_path)

    # -------------------------
    # 14. 预测与评估
    # -------------------------
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    r2_train, mae_train, mse_train, rmse_train = evaluate_regression(y_train, pred_train, 'Train')
    r2_val, mae_val, mse_val, rmse_val = evaluate_regression(y_val, pred_val, 'Validation')
    r2_test, mae_test, mse_test, rmse_test = evaluate_regression(y_test, pred_test, 'Test')

    # -------------------------
    # 过拟合风险评估
    # -------------------------
    r2_gap_val = r2_train - r2_val
    r2_gap_test = r2_train - r2_test
    rmse_ratio_val = rmse_val / rmse_train if rmse_train > 1e-12 else np.inf
    rmse_ratio_test = rmse_test / rmse_train if rmse_train > 1e-12 else np.inf

    print(f"Train-Val R2 gap: {r2_gap_val:.4f}")
    print(f"Train-Test R2 gap: {r2_gap_test:.4f}")
    print(f"Val/Train RMSE ratio: {rmse_ratio_val:.4f}")
    print(f"Test/Train RMSE ratio: {rmse_ratio_test:.4f}")

    val_overfit_strong = (r2_gap_val > 0.10 and rmse_ratio_val > 1.30)
    test_overfit_strong = (r2_gap_test > 0.10 and rmse_ratio_test > 1.30)

    val_overfit_mild = (r2_gap_val > 0.05 and rmse_ratio_val > 1.15)
    test_overfit_mild = (r2_gap_test > 0.05 and rmse_ratio_test > 1.15)

    if val_overfit_strong and test_overfit_strong:
        print("警告：模型在验证集和测试集上均表现出明显过拟合。")
    elif test_overfit_strong:
        print("警告：模型在测试集上表现出明显过拟合。")
    elif val_overfit_strong:
        print("警告：模型在验证集上表现出明显过拟合。")
    elif val_overfit_mild and test_overfit_mild:
        print("提示：模型在验证集和测试集上均存在一定过拟合倾向。")
    elif test_overfit_mild:
        print("提示：模型在测试集上存在一定过拟合倾向。")
    elif val_overfit_mild:
        print("提示：模型在验证集上存在一定过拟合倾向。")
    else:
        print("模型泛化表现正常，未见明显过拟合。")

    if r2_train > 0.98 and (val_overfit_mild or test_overfit_mild):
        print("补充提示：训练集拟合度很高，需进一步警惕过拟合。")

    if val_overfit_strong or test_overfit_strong:
        overfit_summary = "明显过拟合"
    elif val_overfit_mild or test_overfit_mild:
        overfit_summary = "轻微过拟合"
    else:
        overfit_summary = "无明显过拟合"

    print(f"Overfitting assessment: {overfit_summary}")

    # -------------------------
    # 15. 实际值与预测值对比图
    # -------------------------
    plt.figure(figsize=(8, 8), dpi=300)

    plt.plot(y_train, pred_train, 'ko',
             markersize=3,
             label='Train set')

    plt.plot(y_test, pred_test, 'ro',
             markersize=4,
             markerfacecolor='none',
             markeredgewidth=1.2,
             label='Test set')

    plt.tick_params(axis='both', direction='in', length=8, width=2)
    

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.xlabel(r'$\mathit{K}_{\mathrm{t,obs}}[-]$', fontsize=30, fontname='Times New Roman')
    plt.ylabel(r'$\mathit{K}_{\mathrm{t,pred}}[-]$', fontsize=30, fontname='Times New Roman')

    plt.text(
        0.05, 0.96,
    
        f'R$^2$: {r2_test:.4f}\nRMSE: {rmse_test:.4f}',
        transform=plt.gca().transAxes,
        fontsize=22,
        verticalalignment='top',
        horizontalalignment='left',
        fontname='Times New Roman'
    )

    plt.tick_params(labelsize=26)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
      label.set_fontname('Times New Roman')
    plt.plot([0, 1.0], [0, 1.0], '--', color='gray')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    ax.xaxis.set_major_formatter(FuncFormatter(zero_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(zero_formatter))
    #font_cn_legend = FontProperties(family=chosen_font, size=22)
    font_en_legend = FontProperties(
    family='Times New Roman',
    size=22
)
    legend = plt.legend(
      loc='upper right',
      bbox_to_anchor=(1.0, 0.17),
      frameon=False,
      prop=font_en_legend,
      markerscale=2.4,      # 图例中点/图形放大
      handletextpad=0.3,    # 图形和文字之间距离变小
      labelspacing=0.3,     # 各图例条目上下间距
      borderpad=0.2         # 图例内部边距
    )

    plt.tight_layout()
    plt.savefig('train_test_scatter_fast_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

 
# 17. 特征重要性图
 # -------------------------
    booster = model.booster_
    importance = booster.feature_importance(importance_type='split')
    total_splits = np.sum(importance)

    if total_splits == 0:
         importance_ratio = np.zeros_like(importance, dtype=float)
    else:
         importance_ratio = (importance / total_splits) * 100

    feature_names = inputCol
    sorted_idx = np.argsort(importance_ratio)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importance_ratio = importance_ratio[sorted_idx]

    plt.figure(figsize=(12.5, 8), dpi=600)
    ax = plt.gca()

    plt.tick_params(
       axis='both',
       direction='in',
       length=8,
       width=2
    )

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.barh(
        range(len(sorted_importance_ratio)),
        sorted_importance_ratio,
        align='center'
    )

    plt.xlim(
        0,
        max(sorted_importance_ratio) * 1.1
        if max(sorted_importance_ratio) > 0 else 1
    )

    plt.xticks(
        fontsize=28,
        fontname='Times New Roman'
    )

    plt.yticks(
        range(len(sorted_importance_ratio)),
        sorted_names,
        fontsize=20,
        fontname='Times New Roman'
    )

    plt.xlabel(
        'Contribution Degree(%)',
        fontsize=26,
        fontname='Times New Roman'
    )

    plt.title(
        'Feature Importance (FI)',
        fontsize=26,
        fontname='Times New Roman'
    )

    for i, v in enumerate(sorted_importance_ratio):
        plt.text(
            v + 0.2,
            i,
            f"{v:.1f}%",
            fontsize=24,
            fontname='Times New Roman'
     )

    plt.tight_layout()

    plt.savefig(
        'feature_importance_fast_grid.png',
        dpi=600,
        bbox_inches='tight'
     )

    plt.show()

    # -------------------------
    # 18. 保存特征重要性表
    # -------------------------
    feature_importance_df = pd.DataFrame({
        'feature_name': inputCol,
        'importance_split': importance,
        'importance_ratio_percent': importance_ratio
    }).sort_values(by='importance_ratio_percent', ascending=False)

    feature_importance_df.to_csv('feature_importance_fast_grid.csv', index=False, encoding='utf-8-sig')
    print("特征重要性已保存：feature_importance_fast_grid.csv")

    print("程序运行完成。")