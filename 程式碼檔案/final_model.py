#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyproj
from tqdm import tqdm
import folium
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import haversine as hs
from multiprocessing import Pool
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.linear_model import ElasticNet
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

N_FOLD = 20
sub_prefix = 'submission'

random.seed(1994)
np.random.seed(1994)
tqdm.pandas()
pd.set_option('display.max_columns', None)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False # 顯示負號
pd.options.display.float_format = '{:.5f}'.format

start = datetime.datetime.now()


# In[2]:


def catboost_model(x, y, oof_x, oof_y, test_x, categorical_feature, feature_name):
    catboost_params = {
        'learning_rate': 0.03,
        'depth': 8,
        'iterations': 5000,
        'loss_function': 'RMSE',
        'eval_metric': 'MAPE',
        'thread_count': 22,
        'cat_features': categorical_feature,
        'bagging_temperature': 0.95
    }

    y = np.log(y)
    oof_y = np.log(oof_y)
    
    model = CatBoostRegressor(**catboost_params)
    model.fit(x, y, eval_set=(oof_x, oof_y), use_best_model=True, early_stopping_rounds=200, verbose=1000)
    
    oof = model.predict(oof_x)
    preds = model.predict(test_x)
    
    return np.exp(oof), np.exp(preds), model

def xgboost_model(x, y, oof_x, oof_y, test_x, categorical_feature, feature_name):
    xgb_params = {
        'objective': 'reg:squaredlogerror',
        'learning_rate': 0.02,
        'max_depth': 7,
        'n_estimators': 4000,
        'subsample': 0.88,
        'colsample_bytree': 0.55,
        'verbosity': 1,
        'n_jobs': 22,
        'eval_metric': 'mape'
    }
    
    d_train = xgb.DMatrix(x, label=y)
    d_valid = xgb.DMatrix(oof_x, label=oof_y)
    
    model = xgb.train(xgb_params, d_train, num_boost_round=xgb_params['n_estimators'], evals=[(d_valid, 'valid')], early_stopping_rounds=200, feval=None, maximize=False, verbose_eval=1000)
    
    oof = model.predict(d_valid)
    preds = model.predict(xgb.DMatrix(test_x))
    
    return oof, preds, model

def lgb_model(x, y, oof_x, oof_y, test_x, categorical_feature, feature_name):
    lgb_params = {
        'learning_rate': 0.01,
        'application': 'regression',
        'max_depth': 8,
        'num_leaves': 256,
        'feature_fraction': 0.44,
        'bagging_fraction': 0.95,
        'bagging_freq': 8,
        'verbosity': -1,
        'metric': 'mape',
        'num_threads': 22,
        'num_iterations': 7000
    }
    
    y = np.log(y)
    oof_y = np.log(oof_y)
    
    callbacks = [lgb.log_evaluation(period=1000), lgb.early_stopping(stopping_rounds=200)]
    
    d_train = lgb.Dataset(x, label=y)
    d_valid = lgb.Dataset(oof_x, label=oof_y)
    
    model = lgb.train(lgb_params, train_set=d_train, valid_sets=d_valid, callbacks=callbacks, feature_name=feature_name, categorical_feature=categorical_feature)

    oof = model.predict(oof_x)
    preds = model.predict(test_x)
    return np.exp(oof), np.exp(preds), model

def knn_model(x, y, oof_x, oof_y, test_x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    oof_x = scaler.transform(oof_x)
    test_x = scaler.transform(test_x)
    model = neighbors.KNeighborsRegressor(min(7, len(x), len(oof_x)), n_jobs=22)
    model.fit(x, y)
    oof = model.predict(oof_x)
    preds = model.predict(test_x)
    return oof, preds


class mape_tabnet(Metric):
    def __init__(self):
        self._name = "MAPE"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true))
        return mape

def tabnet_model(x, y, oof_x, oof_y, test_x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    oof_x = scaler.transform(oof_x)
    test_x = scaler.transform(test_x)

    # 定義 TabNet 模型
    model = TabNetRegressor(
        #scheduler_params={"step_size":10, "gamma":0.95}
    )
    
    # 訓練模型，提供 early stopping
    model.fit(
        X_train=x, y_train=y.reshape(-1, 1),
        eval_set=[(oof_x, oof_y.reshape(-1, 1))],
        eval_name=["valid"],
        eval_metric=[mape_tabnet],
        max_epochs=200,  # 設定最大迭代次數
        patience=20,  # 設定 early stopping 的耐心度
        batch_size=1024,  # 設定批量大小
        num_workers=22,
        virtual_batch_size=1024,  # 設定虛擬批量大小
    )

    # 預測測試集
    preds = model.predict(test_x).flatten()
    oof = model.predict(oof_x).flatten()
    return oof, preds, model


# In[3]:


def mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape


# In[4]:


df_train = pd.read_csv('../官方資料集/final_feature_engineering_train.csv')
df_valid = pd.read_csv('../官方資料集/final_feature_engineering_valid.csv')


# In[5]:


use_cols = [c for c in df_train.columns if c.isascii()]
use_cols.remove('ID')
use_cols.remove('price')


# In[6]:


def get_corr(a, b):
    _a = []
    _b = []
    for v1, v2 in zip(a, b):
        if v1 > -9999 and v2 > -9999:
            _a.append(v1)
            _b.append(v2)
    return np.corrcoef(_a, _b)[0][1]

def corr_job(cols):
    c1, c2 = cols[0], cols[1]
    corr = get_corr(df_train[c1].values, df_train[c2].values)
    return {
        'key': f'{c1}_{c2}',
        'corr': corr
    }

job_cols = []
for c1 in tqdm(use_cols):
    for c2 in use_cols:
        if c1 == c2:
            continue
        job_cols.append([c1, c2])

with Pool(22) as pool:
    column_corrs = list(tqdm(pool.imap(corr_job, job_cols), total=len(job_cols)))

corr_dict = {}
for d in column_corrs:
    corr_dict.update({
        d['key']: d['corr']
    })
corr_dict


# In[7]:


remove_cols = []

for c1 in tqdm(use_cols):
    for c2 in use_cols:
        if c1 == c2:
            continue
        if c1 in remove_cols:
            continue
        if c2 in remove_cols:
            continue
        key = f'{c1}_{c2}'
        corr = corr_dict[key]
        if corr > 0.999:
            if 'mean' not in c2:
                print(c1, c2, corr)
                remove_cols.append(c2)


# In[8]:


na_cols = []
for c in df_train.columns:
    na_cnt1 = sum(df_train[c].isna())
    na_cnt2 = sum(df_valid[c].isna())
    if na_cnt1 >= len(df_train)*0.8 or na_cnt2 >= len(df_valid)*0.8:
        na_cols.append(c)
        print(c, na_cnt1, na_cnt2)


# In[9]:


print(len(use_cols))
use_cols = [c for c in use_cols if c not in remove_cols]
print(len(use_cols))
use_cols = [c for c in use_cols if c not in na_cols]
print(len(use_cols))
print(use_cols)


# In[10]:


categorical_feature = [
    use_cols.index('city_1'),
    use_cols.index('city_2'),
    use_cols.index('city_3'),
    use_cols.index('city12'),
    use_cols.index('building_type'),
    use_cols.index('main_material'),
    use_cols.index('main_usage'),
    use_cols.index('use_type'),
    use_cols.index('floor_cat'),
    use_cols.index('total_floor_cat'),
    use_cols.index('age_cat'),
]
for i in categorical_feature:
    col = use_cols[i]
    df_train[col] = df_train[col].astype(int)
    df_valid[col] = df_valid[col].astype(int)


# In[11]:


target_col = 'price'


# In[12]:


use_hour = (datetime.datetime.now()-start).total_seconds() / 60 / 60
print(f'準備完成即將開始訓練，目前花費 {use_hour} 小時')


# In[13]:


preds_dict = {
    'knn': {
        'oof': np.zeros(len(df_train)),
        'test': np.zeros(len(df_valid))
    },
    'tabnet': {
        'oof': np.zeros(len(df_train)),
        'test': np.zeros(len(df_valid))
    },
    'xgb': {
        'oof': np.zeros(len(df_train)),
        'test': np.zeros(len(df_valid)),
        'stage1_models': [None for _ in range(N_FOLD)],
        'stage2_models': [None for _ in range(N_FOLD)]
    },
    'lgb': {
        'oof': np.zeros(len(df_train)),
        'test': np.zeros(len(df_valid)),
        'stage1_models': [None for _ in range(N_FOLD)],
        'stage2_models': [None for _ in range(N_FOLD)]
    },
    'cat': {
        'oof': np.zeros(len(df_train)),
        'test': np.zeros(len(df_valid)),
        'stage1_models': [None for _ in range(N_FOLD)],
        'stage2_models': [None for _ in range(N_FOLD)]
    },
}


# In[14]:


preds_dict['lgb']['oof'] = np.zeros(len(df_train))
preds_dict['lgb']['test'] = np.zeros(len(df_valid))

feature_importance_dict = dict(zip(use_cols, [0 for _ in range(len(use_cols))]))

skf = KFold(n_splits=N_FOLD, random_state=23228, shuffle=True)
fold_i = 0
for train_index, test_index in skf.split(df_train):
    fold_i += 1
    print('Start %s fold' % (fold_i))
    _oof, _preds, model = lgb_model(
        df_train[use_cols].fillna(-99999).values[train_index,:], 
        df_train[target_col].values[train_index],
        df_train[use_cols].fillna(-99999).values[test_index,:], 
        df_train[target_col].values[test_index],
        df_valid[use_cols].fillna(-99999).values,
        [],
        use_cols
    )
    preds_dict['lgb']['oof'][test_index] = _oof
    preds_dict['lgb']['test'] += _preds / skf.n_splits
    preds_dict['lgb']['stage1_models'][fold_i-1] = model
    for feature, importance in zip(model.feature_name(), model.feature_importance()):
        feature_importance_dict[feature] += importance

feature_importance_df = pd.DataFrame()
feature_importance_df['Feature'] = list(feature_importance_dict.keys())
feature_importance_df['Importance'] = list(feature_importance_dict.values())
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
feature_importance_df.head(100).sort_values(by='Importance').plot(kind='barh', x="Feature", y="Importance", figsize=(9,32))


# In[15]:


# feature_importance_df.to_csv('feature_importance.csv', index=False)


# In[16]:


score = mape(df_train['price'].values, preds_dict['lgb']['oof'][:len(df_train)])
print(f'{N_FOLD}-Folds CV, LightGBM MAPE = {round(score, 5)}')


# In[17]:


preds_dict['xgb']['oof'] = np.zeros(len(df_train))
preds_dict['xgb']['test'] = np.zeros(len(df_valid))

skf = KFold(n_splits=N_FOLD, random_state=26608, shuffle=True)
fold_i = 0
for train_index, test_index in skf.split(df_train):
    fold_i += 1
    print('Start %s fold' % (fold_i))
    _oof, _preds, model = xgboost_model(
        df_train[use_cols].fillna(-99999).iloc[train_index,:], 
        df_train[target_col].values[train_index],
        df_train[use_cols].fillna(-99999).iloc[test_index,:], 
        df_train[target_col].values[test_index],
        df_valid[use_cols].fillna(-99999),
        categorical_feature,
        use_cols
    )
    preds_dict['xgb']['oof'][test_index] = _oof
    preds_dict['xgb']['test'] += _preds / skf.n_splits
    preds_dict['xgb']['stage1_models'][fold_i-1] = model


# In[18]:


score = mape(df_train['price'].values, preds_dict['xgb']['oof'][:len(df_train)])
print(f'{N_FOLD}-Folds CV, XGBoost MAPE = {round(score, 5)}')


# In[19]:


preds_dict['cat']['oof'] = np.zeros(len(df_train))
preds_dict['cat']['test'] = np.zeros(len(df_valid))

skf = KFold(n_splits=N_FOLD, random_state=23228, shuffle=True)
fold_i = 0
for train_index, test_index in skf.split(df_train):
    fold_i += 1
    print('Start %s fold' % (fold_i))
    _oof, _preds, model = catboost_model(
        df_train[use_cols].fillna(-99999).iloc[train_index,:], 
        df_train[target_col].values[train_index],
        df_train[use_cols].fillna(-99999).iloc[test_index,:], 
        df_train[target_col].values[test_index],
        df_valid[use_cols].fillna(-99999),
        categorical_feature,
        use_cols
    )
    preds_dict['cat']['oof'][test_index] = _oof
    preds_dict['cat']['test'] += _preds / skf.n_splits
    preds_dict['cat']['stage1_models'][fold_i-1] = model


# In[20]:


score = mape(df_train['price'].values, preds_dict['cat']['oof'][:len(df_train)])
print(f'{N_FOLD}-Folds CV, CatBoost MAPE = {round(score, 5)}')


# In[21]:


result = pd.DataFrame()
result['true'] = df_train['price'].values
result['lgb_pred'] = preds_dict['lgb']['oof'][:len(df_train)]
result['xgb_pred'] = preds_dict['xgb']['oof'][:len(df_train)]
result['cat_pred'] = preds_dict['cat']['oof'][:len(df_train)]
result.corr()


# In[22]:


result['ensamble_pred'] = result['lgb_pred']*0.33 + result['cat_pred']*0.33 + result['xgb_pred']*0.34
score = mape(result['true'].values, result['ensamble_pred'])
print(f'{N_FOLD}-Folds CV, Ensemble MAPE = {round(score, 5)}')


# In[23]:


save_cols = ['ID', 'predicted_price']
df_valid['predicted_price'] = preds_dict['cat']['test']*0.34 + preds_dict['lgb']['test']*0.33 + preds_dict['xgb']['test']*0.33
#df_valid[save_cols].to_csv(f'../{sub_prefix}.csv', index=False)


# In[24]:


df_valid[save_cols].describe()


# In[25]:


# stacking
df_oof_preds = pd.DataFrame()
df_oof_preds['price'] = df_train['price']
df_oof_preds['xgb_preds'] = preds_dict['xgb']['oof']
df_oof_preds['lgb_preds'] = preds_dict['lgb']['oof']
df_oof_preds['cat_preds'] = preds_dict['cat']['oof']

df_test_preds = pd.DataFrame()
df_test_preds['xgb_preds'] = preds_dict['xgb']['test']
df_test_preds['lgb_preds'] = preds_dict['lgb']['test']
df_test_preds['cat_preds'] = preds_dict['cat']['test']

model = SVR(kernel='linear', C=0.05, max_iter=2000000)
stacking_cols = ['lgb_preds', 'xgb_preds', 'cat_preds']
x = df_oof_preds[stacking_cols].fillna(0)
pred_x = df_test_preds[stacking_cols].fillna(0)
y = df_oof_preds['price'].values
model.fit(x, y)
df_valid['predicted_price2'] = model.predict(pred_x)
print(df_valid[['predicted_price', 'predicted_price2']].describe())

save_cols = ['ID', 'predicted_price']
df_valid['predicted_price'] = df_valid['predicted_price2']
#df_valid[save_cols].to_csv(f'../{sub_prefix}_with_svr_stacking.csv', index=False)


# In[26]:


use_hour = (datetime.datetime.now()-start).total_seconds() / 60 / 60
print(f'第一階段模型訓練與預測完成，目前花費 {use_hour} 小時')


# In[27]:


df_valid_pseudo = df_valid.copy()
df_valid_pseudo['price'] = df_valid['predicted_price']
df_train_pseudo = pd.concat([df_train, df_valid_pseudo]).reset_index(drop=True)


# In[28]:


df_valid_pseudo['price'].describe()


# In[29]:


cat_epochs = 1.1 * sum(preds_dict['cat']['stage1_models'][i].get_best_iteration() for i in range(len(preds_dict['cat']['stage1_models']))) / len(preds_dict['cat']['stage1_models'])
xgb_epochs = 1.1 * sum(preds_dict['xgb']['stage1_models'][i].best_iteration for i in range(len(preds_dict['xgb']['stage1_models']))) / len(preds_dict['xgb']['stage1_models'])
lgb_epochs = 1.1 * sum(preds_dict['lgb']['stage1_models'][i].best_iteration for i in range(len(preds_dict['lgb']['stage1_models']))) / len(preds_dict['lgb']['stage1_models'])

cat_epochs = int(cat_epochs)
xgb_epochs = int(xgb_epochs)
lgb_epochs = int(lgb_epochs)

print(cat_epochs, xgb_epochs, lgb_epochs)


# In[30]:


def catboost_model(x, y, oof_x, oof_y, test_x, categorical_feature, feature_name):
    catboost_params = {
        'learning_rate': 0.03,
        'depth': 8,
        'iterations': cat_epochs,
        'loss_function': 'RMSE',
        'eval_metric': 'MAPE',
        'thread_count': 22,
        'cat_features': categorical_feature,
        'bagging_temperature': 0.95
    }

    y = np.log(y)
    oof_y = np.log(oof_y)
    
    model = CatBoostRegressor(**catboost_params)
    model.fit(x, y, eval_set=(oof_x, oof_y), use_best_model=True, early_stopping_rounds=200, verbose=1000)
    
    oof = model.predict(oof_x)
    preds = model.predict(test_x)
    
    return np.exp(oof), np.exp(preds), model

def xgboost_model(x, y, oof_x, oof_y, test_x, categorical_feature, feature_name):
    xgb_params = {
        'objective': 'reg:squaredlogerror',
        'learning_rate': 0.02,
        'max_depth': 7,
        'n_estimators': xgb_epochs,
        'subsample': 0.88,
        'colsample_bytree': 0.55,
        'verbosity': 1,
        'n_jobs': 22,
        'eval_metric': 'mape'
    }
    
    d_train = xgb.DMatrix(x, label=y)
    d_valid = xgb.DMatrix(oof_x, label=oof_y)
    
    model = xgb.train(xgb_params, d_train, num_boost_round=xgb_params['n_estimators'], evals=[(d_valid, 'valid')], early_stopping_rounds=200, feval=None, maximize=False, verbose_eval=1000)
    
    oof = model.predict(d_valid)
    preds = model.predict(xgb.DMatrix(test_x))
    
    return oof, preds, model

def lgb_model(x, y, oof_x, oof_y, test_x, categorical_feature, feature_name):
    lgb_params = {
        'learning_rate': 0.01,
        'application': 'regression',
        'max_depth': 8,
        'num_leaves': 256,
        'feature_fraction': 0.44,
        'bagging_fraction': 0.95,
        'bagging_freq': 8,
        'verbosity': -1,
        'metric': 'mape',
        'num_threads': 22,
        'num_iterations': lgb_epochs
    }
    
    y = np.log(y)
    oof_y = np.log(oof_y)
    
    callbacks = [lgb.log_evaluation(period=1000), lgb.early_stopping(stopping_rounds=200)]
    
    d_train = lgb.Dataset(x, label=y)
    d_valid = lgb.Dataset(oof_x, label=oof_y)
    
    model = lgb.train(lgb_params, train_set=d_train, valid_sets=d_valid, callbacks=callbacks, feature_name=feature_name, categorical_feature=categorical_feature)

    oof = model.predict(oof_x)
    preds = model.predict(test_x)
    return np.exp(oof), np.exp(preds), model


# In[31]:


preds_dict['lgb']['oof'] = np.zeros(len(df_train_pseudo))
preds_dict['lgb']['test'] = np.zeros(len(df_valid))
skf = KFold(n_splits=N_FOLD, random_state=23228, shuffle=True)
fold_i = 0
for train_index, test_index in skf.split(df_train_pseudo):
    fold_i += 1
    print('Start %s fold' % (fold_i))
    _oof, _preds, model = lgb_model(
        df_train_pseudo[use_cols].fillna(-99999).values[train_index,:], 
        df_train_pseudo[target_col].values[train_index],
        df_train_pseudo[use_cols].fillna(-99999).values[test_index,:], 
        df_train_pseudo[target_col].values[test_index],
        df_valid[use_cols].fillna(-99999).values,
        [],
        use_cols
    )
    preds_dict['lgb']['oof'][test_index] = _oof
    preds_dict['lgb']['test'] += _preds / skf.n_splits
    preds_dict['lgb']['stage2_models'][fold_i-1] = model


# In[32]:


score = mape(df_train['price'].values, preds_dict['lgb']['oof'][:len(df_train)])
print(f'{N_FOLD}-Folds CV, LightGBM (with pseudo labels) MAPE = {round(score, 5)}')


# In[33]:


preds_dict['cat']['oof'] = np.zeros(len(df_train_pseudo))
preds_dict['cat']['test'] = np.zeros(len(df_valid))

skf = KFold(n_splits=N_FOLD, random_state=21994, shuffle=True)
fold_i = 0
for train_index, test_index in skf.split(df_train_pseudo):
    fold_i += 1
    print('Start %s fold' % (fold_i))
    _oof, _preds, model = catboost_model(
        df_train_pseudo[use_cols].fillna(-99999).iloc[train_index,:], 
        df_train_pseudo[target_col].values[train_index],
        df_train_pseudo[use_cols].fillna(-99999).iloc[test_index,:], 
        df_train_pseudo[target_col].values[test_index],
        df_valid[use_cols].fillna(-99999),
        categorical_feature,
        use_cols
    )
    preds_dict['cat']['oof'][test_index] = _oof
    preds_dict['cat']['test'] += _preds / skf.n_splits
    preds_dict['cat']['stage2_models'][fold_i-1] = model


# In[34]:


score = mape(df_train['price'].values, preds_dict['cat']['oof'][:len(df_train)])
print(f'{N_FOLD}-Folds CV, CatBoost (with pseudo labels) MAPE = {round(score, 5)}')


# In[35]:


preds_dict['xgb']['oof'] = np.zeros(len(df_train_pseudo))
preds_dict['xgb']['test'] = np.zeros(len(df_valid))

skf = KFold(n_splits=N_FOLD, random_state=26608, shuffle=True)
fold_i = 0
for train_index, test_index in skf.split(df_train_pseudo):
    fold_i += 1
    print('Start %s fold' % (fold_i))
    _oof, _preds, model = xgboost_model(
        df_train_pseudo[use_cols].fillna(-99999).iloc[train_index,:], 
        df_train_pseudo[target_col].values[train_index],
        df_train_pseudo[use_cols].fillna(-99999).iloc[test_index,:], 
        df_train_pseudo[target_col].values[test_index],
        df_valid[use_cols].fillna(-99999),
        categorical_feature,
        use_cols
    )
    preds_dict['xgb']['oof'][test_index] = _oof
    preds_dict['xgb']['test'] += _preds / skf.n_splits
    preds_dict['xgb']['stage2_models'][fold_i-1] = model


# In[36]:


score = mape(df_train['price'].values, preds_dict['xgb']['oof'][:len(df_train)])
print(f'{N_FOLD}-Folds CV, XGBoost (with pseudo labels) MAPE = {round(score, 5)}')


# In[37]:


preds_dict['tabnet']['oof'] = np.zeros(len(df_train_pseudo))
preds_dict['tabnet']['test'] = np.zeros(len(df_valid))

skf = KFold(n_splits=N_FOLD, random_state=26608, shuffle=True)
fold_i = 0
for train_index, test_index in skf.split(df_train_pseudo):
    fold_i += 1
    print('Start %s fold' % (fold_i))
    _oof, _preds, model = tabnet_model(
        df_train_pseudo[use_cols].fillna(df_train_pseudo[use_cols].mean()).iloc[train_index,:], 
        df_train_pseudo[target_col].values[train_index],
        df_train_pseudo[use_cols].fillna(df_train_pseudo[use_cols].mean()).iloc[test_index,:], 
        df_train_pseudo[target_col].values[test_index],
        df_valid[use_cols].fillna(df_valid[use_cols].mean())
    )
    preds_dict['tabnet']['oof'][test_index] = _oof
    preds_dict['tabnet']['test'] += _preds / skf.n_splits


# In[38]:


score = mape(df_train['price'].values, preds_dict['tabnet']['oof'][:len(df_train)])
print(f'{N_FOLD}-Folds CV, TabNet (with pseudo labels) MAPE = {round(score, 5)}')


# In[39]:


prices_cols = ['nearest_100_price_mean',
 'nearest_30_price_mean',
 'nearest_10_price_mean',
 'city12_price_mean',
 'externalkey_price_mean',
 'externalkey_samebuilding_price_mean',
 'externalkey_samefloor_price_mean',
 'externalkey_sameage_price_mean',
 'externalkey_sameage_05_price_mean',
 'externalkey_sameage_0.25_price_mean',
 'externalkey_exactly_same_price_mean']


# In[40]:


preds_dict['knn']['oof'] = np.zeros(len(df_train_pseudo))
preds_dict['knn']['test'] = np.zeros(len(df_valid))

skf = KFold(n_splits=N_FOLD, random_state=26608, shuffle=True)
fold_i = 0
for train_index, test_index in skf.split(df_train_pseudo):
    fold_i += 1
    print('Start %s fold' % (fold_i))
    _oof, _preds = knn_model(
        df_train_pseudo[prices_cols].fillna(df_train_pseudo[prices_cols].mean()).iloc[train_index,:], 
        df_train_pseudo[target_col].values[train_index],
        df_train_pseudo[prices_cols].fillna(df_train_pseudo[prices_cols].mean()).iloc[test_index,:], 
        df_train_pseudo[target_col].values[test_index],
        df_valid[prices_cols].fillna(df_valid[prices_cols].mean())
    )
    preds_dict['knn']['oof'][test_index] = _oof
    preds_dict['knn']['test'] += _preds / skf.n_splits


# In[41]:


score = mape(df_train['price'].values, preds_dict['knn']['oof'][:len(df_train)])
print(f'{N_FOLD}-Folds CV, KNN (with pseudo labels) MAPE = {round(score, 5)}')


# In[42]:


result = pd.DataFrame()
result['true'] = df_train['price'].values
result['lgb_pred'] = preds_dict['lgb']['oof'][:len(df_train)]
result['xgb_pred'] = preds_dict['xgb']['oof'][:len(df_train)]
result['cat_pred'] = preds_dict['cat']['oof'][:len(df_train)]
result['tabnet_pred'] = preds_dict['tabnet']['oof'][:len(df_train)]
result['knn_pred'] = preds_dict['knn']['oof'][:len(df_train)]

print(result.describe())
print(result.corr())


# In[43]:


result['ensamble_pred'] = result['lgb_pred']*0.33 + result['cat_pred']*0.33 + result['xgb_pred']*0.34
score = mape(result['true'].values, result['ensamble_pred'])
print(f'{N_FOLD}-Folds CV, Ensemble (with pseudo labels) MAPE = {round(score, 5)}')


# In[44]:


save_cols = ['ID', 'predicted_price']
df_valid['predicted_price'] = preds_dict['cat']['test']*0.34 + preds_dict['lgb']['test']*0.33 + preds_dict['xgb']['test']*0.33
#df_valid[save_cols].to_csv(f'../{sub_prefix}_pseudo.csv', index=False)


# In[45]:


df_valid[save_cols].describe()


# In[46]:


# stacking
df_oof_preds = pd.DataFrame()
df_oof_preds['price'] = df_train['price']
df_oof_preds['xgb_preds'] = preds_dict['xgb']['oof'][:len(df_train)]
df_oof_preds['lgb_preds'] = preds_dict['lgb']['oof'][:len(df_train)]
df_oof_preds['cat_preds'] = preds_dict['cat']['oof'][:len(df_train)]
df_oof_preds['tabnet_preds'] = preds_dict['tabnet']['oof'][:len(df_train)]
df_oof_preds['knn_preds'] = preds_dict['knn']['oof'][:len(df_train)]

df_test_preds = pd.DataFrame()
df_test_preds['xgb_preds'] = preds_dict['xgb']['test']
df_test_preds['lgb_preds'] = preds_dict['lgb']['test']
df_test_preds['cat_preds'] = preds_dict['cat']['test']
df_test_preds['tabnet_preds'] = preds_dict['tabnet']['test']
df_test_preds['knn_preds'] = preds_dict['knn']['test']

model = SVR(kernel='linear', C=0.1, max_iter=2000000)
stacking_cols = ['lgb_preds', 'xgb_preds', 'cat_preds', 'tabnet_preds', 'knn_preds']
x = df_oof_preds[stacking_cols].fillna(0)
pred_x = df_test_preds[stacking_cols].fillna(0)
y = df_oof_preds['price'].values
model.fit(x, y)
df_valid['predicted_price2'] = model.predict(pred_x)
print(df_valid[['predicted_price', 'predicted_price2']].describe())

save_cols = ['ID', 'predicted_price']
df_valid['predicted_price'] = df_valid['predicted_price2']
df_valid[save_cols].to_csv(f'../{sub_prefix}.csv', index=False)


# In[47]:


use_hour = (datetime.datetime.now()-start).total_seconds() / 60 / 60
print(f'第二階段預測完成，總花費 {use_hour} 小時')


# In[ ]:




