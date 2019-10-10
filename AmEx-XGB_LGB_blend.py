# -*- coding: utf-8 -*-
"""

AM-Expert Hackathon - Data prep. and analysis

------------------------
Created: Oct 6, 2019

@author: IME
"""


### import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import tensorflow as tf
import xgboost    as xgb
import lightgbm   as lgb

from xgboost               import XGBClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import VotingClassifier
from sklearn.linear_model  import LogisticRegression, SGDClassifier
from sklearn.tree          import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.naive_bayes   import GaussianNB

from sklearn.model_selection  import train_test_split
from sklearn.model_selection  import KFold, StratifiedKFold
from sklearn.model_selection  import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection  import cross_val_score
from sklearn.model_selection  import learning_curve
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

#sorted(sklearn.metrics.SCORERS.keys())

#%% Helper functions

def submit(predictions, file_name):
    df_subm = pd.read_csv('./inputs/sample_submission.csv')
    df_subm['redemption_status'] = predictions
    df_subm.to_csv((('%s.csv') % (file_name)), index=False)


def data_split(df_all):
    """
     Split the combined df to train/test and separate target var. and ID cols
     Inputs:
     :param df_all: combined dataframe with all features and target
     Outputs:
     :return: train df, test df
    """
    # split to train/test
    df_train = df_all.loc[df_all.source == 'train', :]
    df_test = df_all.loc[df_all.source == 'test', :]

    df_train = df_train.drop(['source'], axis=1)
    df_test = df_test.drop(['source'], axis=1)
    # drop target var from test set
    df_test = df_test.drop(['redemption_status'], axis=1)
    # target and ID cols
    target = 'redemption_status'
    id_cols = ['id', 'campaign_id', 'coupon_id', 'customer_id']
    return df_train, df_test, target, id_cols

#%% Load data
data_all = pd.read_csv("./data/data_encoded_4.csv", header=0, sep=',', decimal='.')

# split to train/test 
data_train, data_test, target_var, id_cols = data_split(data_all)

# save targets
targets = data_train[target_var]

# train cols - predictors
features_all = [f for f in data_train.columns if f not in [target_var]+['id']]

features = [
    'campaign_id', 'coupon_id', 'customer_id', # 'id', 'redemption_status',
    'item_id_x', # 'item_id_y',                # 'quantity',
    'brand',
    'campaign_x',                              # 'campaign_y',
    'brand_type_est',
    'brand_type_loc',
    'Married',  # 'Single',
    'rented_1',
    'campaign_duration',
    'trans_duration',
    'selling_price_log',
    'other_discount_log',
    'coupon_discount_log',
     #'category_code',
    'categ_code_1', 'categ_code_2', 'categ_code_3', 'categ_code_4', 'categ_code_5',
    'categ_code_6', 'categ_code_7', 'categ_code_8', 'categ_code_9', 'categ_code_10',
     #'family_size_code',
    'family_code_1', 'family_code_2', 'family_code_3', 'family_code_4',
     #'age_range_code',
    'age_range_code_1', 'age_range_code_2', 'age_range_code_3', 'age_range_code_4', 'age_range_code_5',
    # 'income_bracket_code',
    'income_code_1', 'income_code_2', 'income_code_3', 'income_code_4', 'income_code_5',
    'income_code_6', 'income_code_7', 'income_code_8', 'income_code_9', 'income_code_10', 'income_code_11',
    #'no_of_children_code',
    'no_child_code_1', 'no_child_code_2', 'no_child_code_3']

#%% Split to train/val sets
X_train, X_val, y_train, y_val = train_test_split(data_train[features], data_train[target_var], 
                                                  test_size=0.25, shuffle=True, random_state=26) 

print('Class distributions - Train: \n', y_train.value_counts(normalize=True))
print('Class distributions - Val: \n', y_val.value_counts(normalize=True))

# ALL data
X_train_all = data_train[features]
y_train_all = data_train[target_var]

print('')
print('Class-1 distribution (%) - all:', y_train_all.mean()*100)  # 0.9 %

# Test data
X_test = data_test[features]

# del X_train, X_val, y_train, y_val, X_train_all, y_train_all, X_test

#%% XGB 
xgb_params = dict(
   learning_rate=0.01,     # 0.07
   n_estimators=1000,
   max_depth=8,
   min_child_weight=3,
   subsample=0.9,          # try 0.8
   colsample_bytree=0.9,
   objective='binary:logistic',
   booster='gbtree',
   scale_pos_weight=50,
   gamma=0.5,              # 0.0
   reg_alpha=0.1 )

xgb_clf = xgb.XGBClassifier(**xgb_params)

# cv with 10-Fold
# kf = KFold(n_splits=10, random_state=26, shuffle=True)
# cv_score = cross_val_score(xgb_clf, X_train, y_train, cv=kf, scoring='roc_auc')
# print(cv_score, np.mean(cv_score))

# # XGB fit for tuning
# xgb_clf.fit(X_train, y_train, early_stopping_rounds=100,
#             eval_metric='auc',
#             eval_set=[(X_train, y_train), (X_val, y_val)],
#             verbose=100)

cvTrain = True     # True: for CV using XGB CV

if cvTrain == True:    
    # cv using XGB API
    cvXGB = xgb.cv(xgb_params, xgb.DMatrix(X_train_all, label=y_train_all), nfold=10, 
                   metrics='auc',
                   num_boost_round=xgb_clf.get_params()['n_estimators'],  
                   early_stopping_rounds=100)
    
    # set no. estimators
    xgb_clf.set_params(n_estimators=cvXGB.shape[0])


# XGB fit with ALL data
xgb_clf.fit(X_train_all, y_train_all, eval_metric='auc')
    
# XGB predict
xgb_pred_tr = xgb_clf.predict_proba(X_train)[:, 1]
xgb_pred_val = xgb_clf.predict_proba(X_val)[:, 1]

# XGB results
print(confusion_matrix(y_val, xgb_clf.predict(X_val)))
print('')
print('train AUC = %0.4f' % roc_auc_score(y_train, xgb_pred_tr))
print('val AUC = %0.4f' % roc_auc_score(y_val, xgb_pred_val))


#%% Imp. Features XGB
imp_features_df = pd.DataFrame()
imp_features_df['Feature'] = X_train.columns
imp_features_df['Importance'] = xgb_clf.feature_importances_
imp_features_df.sort_values(by=['Importance'], ascending=False, inplace=True)

plt.figure()
sns.barplot(imp_features_df['Feature'], imp_features_df['Importance'])
plt.xticks(rotation=90)
plt.show()

# select top features
xgb_features = imp_features_df['Feature'].head(15)   # 20

data_train[xgb_features].head()

# -------------------------------------------------------
# #%%% XGB Grid-search
#
# params = {
#         'learning_rate': [0.05, 0.1],
#         'n_estimators': [50, 100, 300],
#         'max_depth': [7, 8, 9],
#         'colsample_bytree': [0.8, 0.9, 1.0]
#         }
#         # 'reg_alpha': [0.3, 0.4, 0.5],
#         # 'objective': ['binary:logistic']
#
# # initialize XGB
# xgb_clf = xgb.XGBClassifier()
#
# # Grid-Search
# gs = GridSearchCV(estimator=xgb_clf, param_grid=params,
#                   cv=5, verbose=True, scoring='roc_auc')
#
# #gs = RandomizedSearchCV(estimator=xgb_clf0, param_distributions=params,
# #                        cv=5, verbose=1, scoring=gs_score, n_iter=5)
#
# gs.fit(X_train, y_train)
#
# # Display best score and params
# print('Best score:', gs.best_score_)
# # Random GS -->
# # GS -->
#
# print('Best Params:', gs.best_params_)
# ----------------------------------------------

#%% LGB

err = []
y_pred_tot = []

kf = KFold(n_splits=10, shuffle=True, random_state=26)
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=26)

i = 1
for train_index, test_index in kf.split(X_train_all, y_train_all):
    X_train, X_test = X_train_all.iloc[train_index], X_train_all.iloc[test_index]
    y_train, y_test = y_train_all[train_index], y_train_all[test_index]

    lgbm = lgb.LGBMClassifier(n_estimators=1000,
                              learning_rate=0.07,
                              boosting_type='gbdt',
                              num_leaves=31,
                              max_depth=-1,
                              min_child_weight=0.01,
                              colsample_bytree=0.9,
                              random_state=26)

    lgbm.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             eval_metric='auc',
             early_stopping_rounds=100,
             verbose=100)

    preds_oof = lgbm.predict_proba(X_test)[:, -1]

    print("ROC_AUC Score: ", roc_auc_score(y_test, preds_oof))
    err.append(roc_auc_score(y_test, preds_oof))

    p = lgbm.predict_proba(data_test[features])[:, -1]
    print(f'--------------------Fold {i} completed !!!------------------')
    i = i + 1
    y_pred_tot.append(p)

# 10-fold mean prediction
y_pred_lgb = np.mean(y_pred_tot, 0)

plt.figure()
plt.hist(y_pred_lgb, bins=50)
plt.show()


#%% PREDICT Test set

# LGB + XGB blending
# -----------------
y_pred_xgb = xgb_clf.predict_proba(data_test[features])[:, 1]

df_subm_cor = pd.DataFrame()
df_subm_cor['xgb'] = y_pred_xgb
df_subm_cor['lgb'] = y_pred_lgb

df_subm_cor.corr()
# xgb  1.000000  0.654111
# lgb  0.654111  1.000000

plt.figure()
df_subm_cor['xgb'].hist(bins=50)
df_subm_cor['lgb'].hist(bins=50)
plt.show()
# ------------------------------------

# blend predictions
pred_test_avg = 0.5*y_pred_xgb + 0.5*y_pred_lgb

# submit predictions
submit(pred_test_avg, 'submision_xgb-lgb')

# Public LB:0.8619
# Pvt LB:0.87001
