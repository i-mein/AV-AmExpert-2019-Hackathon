"""

AM-Expert Hackathon - Data prep. and analysis

------------------------
Created: Oct 5, 2019

@author: IME
"""

#%%
import numpy as np 
import pandas as pd 
import seaborn as sns 
import datetime as dt
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble      import RandomForestClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection  import train_test_split 
from sklearn.model_selection  import KFold 
from sklearn.model_selection  import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import scipy
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
import gc
gc.enable()


import tensorflow as tf
import xgboost    as xgb
import lightgbm   as lgb

#%% Helper functions

def submit(predictions, file_name):
    df_subm = pd.read_csv('./inputs/sample_submission.csv')
    df_subm['redemption_status'] = predictions
    df_subm.to_csv((('%s.csv') % (file_name)), index=False)

#%% Load raw data

# train file
data_train = pd.read_csv("./inputs/train.csv", header=0, sep=',', decimal='.')

# test file
data_test = pd.read_csv("./inputs/test.csv", header=0, sep=',', decimal='.')

# Link: train --> campaign_data, on = 'campaign_id'
data_campaign = pd.read_csv("./inputs/campaign_data.csv", header=0, sep=',', decimal='.')

# Link: train --> Coupon_item_mapping, on ='item_id'
data_coupon = pd.read_csv("./inputs/coupon_item_mapping.csv", header=0, sep=',', decimal='.')

# Link: Coupon_item_mapping --> item_data, on ='item_id'
data_item = pd.read_csv("./inputs/item_data.csv", header=0, sep=',', decimal='.')

# Link:  train --> customer_demographics, on ='customer_id'
data_customer_demo = pd.read_csv("./inputs/customer_demographics.csv", header=0, sep=',', decimal='.')

# Link:  train --> customer_transaction_data, on ='customer_id'
data_customer_trans = pd.read_csv("./inputs/customer_transaction_data.csv", header=0, sep=',', decimal='.')

# Combine Train/Test data
data_train['source'] = 'train'
data_test['source'] = 'test'
data_all = pd.concat([data_train, data_test], axis=0, ignore_index=True) 

print('Shape train/test:', data_train.shape, data_test.shape)
print('Shape combined:', data_all.shape)
print('-' * 30)
print('campaign data shape', data_campaign.shape)
print('-' * 30)
print('data coupon shape', data_coupon.shape)
print('-' * 30)
print('data item shape', data_item.shape)
print('-' * 30)

no_train_samples = data_train.shape[0]

#%% Train data info
# -------------------
data_train.info()

print('Check for null values..')   # OK
data_train.isnull().sum()

# save target var
target_var = 'redemption_status'
targets = data_train[target_var]

print('Class distributions:\n', targets.value_counts(normalize=True))
# Class-0    77640  -- 99.07%
# Class-1      729  --  0.93%


# Train.id
data_train.id.nunique()          #  78369  -- ALL unique!
data_test.id.nunique()           #  50226
#
print('IDs appear only in train set: \n')
set(data_train.id.unique()).difference(set(data_test.id.unique() ))

print('IDs common in train/test: \n')
set(data_train.id.unique()).intersection(set(data_test.id.unique() ))

# Train.campaign_id
data_train.campaign_id.nunique()           # 18

campaign_id_train = np.sort(data_train.campaign_id.unique())   # {1-13, 26-30}
campaign_id_train = set(campaign_id_train)

data_train.campaign_id.value_counts()

plt.figure(figsize=(8, 6))
sns.countplot(x='campaign_id', data=data_train)
plt.title('Train-campaign_id', fontsize=16)
plt.show()

# Train.coupon_id
data_train.coupon_id.nunique()           # 866 
coupon_id_train = np.sort(data_train.coupon_id.unique())
coupon_id_train = set(coupon_id_train)

# Train.customer_id
data_train.customer_id.nunique()         # 1428


# """
# # Merge procedure:
# -----------------------
# TODO: Merge train + campaign  --> OK
# * data_all + campaign_data
#
# TODO: Merge Coupon + Item  --> OK
# * coupon_item_mapping + item_data --> coupon_item, on='item_id'
#
# TODO: Merge train + Coupon-Item
# * data_all + coupon_item
#
# TODO: Merge customer demo + customer trans
# * data_customer_demo + data_customer_trans --> data_customer, on ='customer_id'
#
# TODO: Merge train + customer
# * data_all + data_customer
# -----------------------
# """


#%% Campaing data 

data_campaign.info()
data_campaign.isnull().sum()

# data_campaign.head()

# campaign_data.campaign_id
data_campaign.campaign_id.nunique()    # 28
np.sort(data_campaign.campaign_id.unique())

# campaign_ids in Test
#[16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


# campaign_type
data_campaign.campaign_type.nunique()          # 2
data_campaign.campaign_type.value_counts()
# Y:   22
# X:    6


# start/end dates
data_campaign['start_date'] = pd.to_datetime(data_campaign.start_date)
data_campaign['end_date'] = pd.to_datetime(data_campaign.end_date)

# make a new copy for preprocessing
data_merged = data_all.copy()

### Merge with campaign data
data_merged = data_merged.merge(data_campaign, how='left', on='campaign_id')

# TODO: FE Campaign_data - duration, month, year etc. --> OK

#%% Coupon Item || Item

# Merge  Coupon_Item_Mapping  with Item Data, on='item_id'

data_coupon.info()
data_coupon.isnull().sum()     # OK

data_item.info()
data_item.isnull().sum()       # OK

# coupon.coupon_id
data_coupon.coupon_id.nunique()     # 1116

data_train.coupon_id.nunique()      # 866
data_test.coupon_id.nunique()       # 331
data_all.coupon_id.nunique()        # 1116
data_merged.coupon_id.nunique()     # 1116

plt.figure()
sns.countplot(x='coupon_id', data=data_coupon)
plt.title('Coupon-coupon_id', fontsize=16)
plt.show()


# coupon.item_id
data_coupon.item_id.nunique()       # 36289 unique items
data_coupon.item_id.describe()

# data_item.item_id
data_item.item_id.nunique()         # 74066 unique items
data_item.item_id.describe()

### Merge Coupon | Item
data_coupon = data_coupon.merge(data_item, how='left', on='item_id')

print('Merged Coupon + Item data -- OK')
data_coupon.info()         # 92663 x 5



# Brand
data_coupon.brand.nunique()          # 2555 brands

plt.figure()
sns.countplot(x='brand', data=data_coupon)
plt.title('Coupon-brand', fontsize=16)
plt.show()

# Brand_type
data_coupon.brand_type.nunique()     # 2 brand types
data_coupon.brand_type.value_counts()
# Established    78759
# Local          13904

plt.figure()
sns.countplot(x='brand_type', data=data_coupon)
plt.title('Coupon-brand_type', fontsize=16)
plt.show()

# category (item)
data_coupon.category.nunique()       # 17 categories
data_coupon.category.value_counts()

plt.figure()
sns.countplot(x='category', data=data_coupon)
plt.title('Coupon-category', fontsize=16)
plt.xticks(rotation=90)
plt.show()

# Grocery                   36466
# Pharmaceutical            25061
# Natural Products           6819
# Meat                       6218
# Packaged Meat              6144
# Skin & Hair Care           4924
# Seafood                    2227
# Flowers & Plants           1963
# Dairy, Juices & Snacks     1867
# Garden                      286
# Prepared Food               240
# Miscellaneous               184
# Salads                      100
# Bakery                      100
# Travel                       44
# Vegetables (cut)             19
# Restauarant                   1


# group data_coupon wrt coupon_id so every id appears once  -- group by mode
data_coupon_group = data_coupon.groupby('coupon_id').agg(lambda x: scipy.stats.mode(x)[0])
# data_coupon.groupby('coupon_id').agg(pd.Series.mode)

print('data_coupon reduced using groupby.. \n')
print('data coupon group shape: ', data_coupon_group.shape)
print('with unique coupon ids:', data_coupon_group.index.nunique())

data_coupon_group = data_coupon_group.reset_index()


### Merge Train | Coupon + Item
data_merged = data_merged.merge(data_coupon_group, how='left', on='coupon_id')

# # export to csv
# data_merged.to_csv('./data/data_coupon_campaign.csv', index=None)


#%% Customer datasets

data_customer_demo.info()
data_customer_trans.info()

print('customer demo shape:', data_customer_demo.shape)        # 760 x 7
print('customer trans shape', data_customer_trans.shape)       # 1324566 x 7

print('customer demo null values')
data_customer_demo.isnull().sum()
# ---------------------
# customer_id         0
# age_range           0
# marital_status    329
# rented              0
# family_size         0
# no_of_children    538
# income_bracket      0
# ---------------------
# TODO: fill in missing values data_customer_demo.marital_status + data_customer_demo.family_size

family_child_miss = data_customer_demo.loc[data_customer_demo.no_of_children.isnull(), :][['family_size', 'no_of_children']]
# (538, 2)

family_child_miss.family_size.value_counts()

# Replace missing values for 'no_of_children' --> '0'
# data_customer_demo.loc[data_customer_demo.no_of_children.isnull(), :]['no_of_children']
data_customer_demo['no_of_children'] = data_customer_demo['no_of_children'].replace(to_replace=np.nan, value=0)
data_customer_demo['no_of_children'] = data_customer_demo['no_of_children'].replace(to_replace='3+', value=3)

data_customer_demo['no_of_children'].value_counts()
data_customer_demo.loc[(data_customer_demo['marital_status']==np.nan) & (data_customer_demo['no_of_children']>='1')]

# Replace marital status NaN with 'unknown' status
data_customer_demo['marital_status'] = data_customer_demo['marital_status'].replace(to_replace=np.nan, value='Unknown')

data_customer_demo['marital_status'].value_counts()
# Unknown    329
# Married    317
# Single     114

# Replace family_size '5+'  with '5' category
data_customer_demo['family_size'] = data_customer_demo['family_size'].replace(to_replace='5+', value='5')
data_customer_demo['family_size'].value_counts()


data_customer_demo['age_range'].value_counts()
# 46-55    271
# 36-45    187
# 26-35    130
# 70+       68
# 56-70     59
# 18-25     45

data_customer_demo['income_bracket'].value_counts()
# 5     187
# 4     165
# 6      88
# 3      70
# 2      68
# 1      59
# 8      37
# 7      32
# 9      29
# 12     10
# 10     10
# 11      5
# -------------------------------------------------

print('customer trans null values')
data_customer_trans.isnull().sum()     # OK

# customer_id
data_customer_demo.customer_id.nunique()     # 760 unique customer ids  [1 - 1581]
data_customer_demo.customer_id.describe()

data_customer_trans.customer_id.nunique()    # 1582 unique customer ids
data_customer_trans.customer_id.describe()

data_merged.customer_id.nunique()            # 1582 unique customer ids
data_merged.customer_id.describe()

# find common customer ids
len(set(data_customer_trans.customer_id).intersection(set(data_customer_demo.customer_id)))

# group customer_trans by customer id
data_customer_trans.groupby('customer_id').agg(lambda x: scipy.stats.mode(x)[0])

data_customer_group = data_customer_trans.groupby('customer_id').agg({
                                                'date': lambda x: scipy.stats.mode(x)[0],
                                                'item_id': lambda x: scipy.stats.mode(x)[0],
                                                'quantity': lambda x: scipy.stats.mode(x)[0],
                                                'selling_price': 'mean',
                                                'other_discount': 'mean',
                                                'coupon_discount': 'mean'})

data_customer_group = data_customer_group.reset_index()

# TODO: rename col. date --> 'trans_date'

data_customer_group.customer_id.nunique()      # 1582

data_customer_group.isnull().sum()             # OK

### Merge customer_demo + customer_group (customer_trans)
data_customer = data_customer_group.merge(data_customer_demo, how='left', on='customer_id').fillna(method='ffill')

data_customer['no_of_children'] = data_customer['no_of_children'].replace(to_replace=np.nan, value=0)


### Merge with data_customer_demo-without-NaNs
data_customer_nan = data_customer_group.merge(data_customer_demo, how='left', on='customer_id')


print('merged customer data shape:', data_customer.shape)   # 1582 x 13

print('merged customer data info:', data_customer.info())

print('merged customer data null:\n', data_customer.isnull().sum())
# ------------------------     ffill     data_customer_nan
# customer_id           0        0             0
# date                  0        0             0
# item_id               0        0             0
# quantity              0        0             0
# selling_price         0        0             0
# other_discount        0        0             0
# coupon_discount       0        0             0
# age_range           822        0            822
# marital_status     1151        0            822
# rented              822        0            822
# family_size         822        0            822
# no_of_children     1360        6-->0        822
# income_bracket      822        0            822
# ------------------------------------------------------
# TODO: fill in missing values data_customer_nan


### Merge data_customer | train
data_merged_1 = data_merged.merge(data_customer, how='left', on='customer_id')

# data_merged_2 = data_merged.merge(data_customer_nan, how='left', on='customer_id')

# # save to csv Merged data v.1
# data_merged_1.to_csv('./data/data_merged_all_1.csv', index=None)

#%% Merged data analysis and encoding

print('All data merged v.1 shape: ', data_merged_1.shape)
data_merged_1.info()

# TODO: drop item_id_x, item_id_y  ??

# save a copy before further processing..
data = data_merged_1.copy()

# change dtype of target var.
# data['redemption_status'] = data['redemption_status'].astype(np.int16)

# change dtype and name of 'date'
data['trans_date'] = pd.to_datetime(data['date'])
data = data.drop(['date'], axis=1)

# OHE campaign_type
data = pd.concat([data, pd.get_dummies(data['campaign_type'])], axis=1)      # drop_first=True
data = data.drop(['campaign_type'], axis=1)
# rename cols
data = data.rename(columns={'X': 'campaign_x', 'Y': 'campaign_y'})


# OBJECT cols
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

#  'source',        --> drop
#  'brand_type',     --> OHE
#  'category',       --> LE
#  'age_range',      --> LE
#  'marital_status', --> OHE
#  'family_size',    --> LE
#  'no_of_children'  --> LE

data['brand_type'].value_counts()     # 2-levels
# Established    103622
# Local           24973

data['category'].value_counts()       # 11-levels
# Grocery                   86782
# Pharmaceutical            29867
# Packaged Meat              2896
# Dairy, Juices & Snacks     1811
# Seafood                    1723
# Natural Products           1589
# Prepared Food              1572
# Bakery                      758
# Skin & Hair Care            702
# Meat                        638
# Flowers & Plants            257

data['age_range'].value_counts()       # 6-levels
# 46-55    45343
# 36-45    33983
# 26-35    22963
# 70+      10925
# 56-70     8743
# 18-25     6638

data['marital_status'].value_counts()  # 2-levels
# Married    94392
# Single     34203

data['family_size'].value_counts()     # 5-levels
# 2     50596
# 1     39416
# 3     19361
# 5+    10172
# 4      9050

data['no_of_children'].value_counts()  # 4-levels
# 1     58773
# 2     35454
# 3+    33965
# 0       403

#%% Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['category_code'] = le.fit_transform(data['category'])
data = data.drop(['category'], axis=1)


data = pd.concat([data, pd.get_dummies(data['brand_type'])], axis=1)      # drop_first=True
data = data.drop(['brand_type'], axis=1)
# rename cols
data = data.rename(columns={'Established': 'brand_type_est', 'Local': 'brand_type_loc'})


le = LabelEncoder()
data['age_range_code'] = le.fit_transform(data['age_range'])
data = data.drop(['age_range'], axis=1)


data = pd.concat([data, pd.get_dummies(data['marital_status'])], axis=1)      # drop_first=True
data = data.drop(['marital_status'], axis=1)


le = LabelEncoder()
data['family_size_code'] = le.fit_transform(data['family_size'])
data = data.drop(['family_size'], axis=1)


data['no_of_children'] = data['no_of_children'].replace(to_replace='3+', value='3')
data['no_of_children'] = data['no_of_children'].replace(to_replace=0, value='0')
# data['no_of_children'] = data['no_of_children'].astype('category')

le = LabelEncoder()
data['no_of_children_code'] = le.fit_transform(data['no_of_children'].values)
data = data.drop(['no_of_children'], axis=1)


data['income_bracket'] = data['income_bracket'].astype('category')
le = LabelEncoder()
data['income_bracket_code'] = le.fit_transform(data['income_bracket'].values)
data = data.drop(['income_bracket'], axis=1)

data = pd.concat([data, pd.get_dummies(data['rented'], drop_first=True, prefix='rented')], axis=1)      #
data = data.drop(['rented'], axis=1)

data = data.rename(columns={'rented_1.0': 'rented_1'})

data.to_csv('./data/data_encoded_1.csv', index=None)

#%% Encoding date cols

data_all = pd.read_csv("./data/data_encoded_1.csv", header=0, sep=',', decimal='.')
print('Processed datased loaded! \n')

data_all.info()

# -------------------
# TODO: FE on dates!
# -------------------
data_all['start_date'] = pd.to_datetime(data_all['start_date'])
data_all['end_date'] = pd.to_datetime(data_all['end_date'])
data_all['trans_date'] = pd.to_datetime(data_all['trans_date'])
#
#data_all['start_date'].dt.year 
#
#data_all['end_date'] - data_all['start_date'] 
#
#data_all['end_date'][0] - data_all['start_date'][0]   # -12 days 
#data_all['end_date'][2] - data_all['start_date'][2]   # 31 days
#
print('strange campaign start/end dates')
data_all.loc[(data_all['end_date'] - data_all['start_date'])<pd.Timedelta(0), :][['start_date', 'end_date']]
## 54018 rows x 2 columns

print('strange campaign trans/start-end dates')
(data_all['trans_date'] - data_all['start_date'])<pd.Timedelta(0)


# new timedelta features
data_all['campaign_duration'] = np.abs((data_all['end_date'] - data_all['start_date']).dt.days).astype(np.int) 
data_all['trans_duration'] = np.abs((data_all['trans_date'] - data_all['start_date']).dt.days).astype(np.int) 

# drop date cols
data_all = data_all.drop(['start_date', 'end_date', 'trans_date'], axis=1)

# ### export to csv
# data_all.to_csv('./data/data_encoded_2.csv', index=None)

#%% Data exploratory
# -------------------

# quanity  --> drop
plt.figure()
data_all['quantity'].hist()
plt.show()
data_all['quantity'].value_counts()


# selling_price
fig, ax = plt.subplots(1, 2)
ax[0].hist(data_all['selling_price'], bins=50, label='orig')
ax[0].legend()
ax[1].hist(np.log1p(data_all['selling_price']), bins=50, label='log')
ax[1].legend()
plt.suptitle('selling price')
plt.show()

# np.log1p(data_train['selling_price']).skew()  # 0.449
# data_train['selling_price'].skew()            # 2.302
# data_train['selling_price'].quantile(0.9)     # 150.00
# data_train['selling_price'].describe()

# other_discount
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].hist(data_all['other_discount'], bins=50, label='original')
ax[0].legend()
ax[1].hist(np.log1p(data_all['other_discount']**2), bins=50, label='squared+log')
ax[1].legend()
# ax[1].hist(np.log1p(data_all['other_discount']), bins=50)
plt.suptitle('other discount')
plt.show()

data_all['other_discount'].describe()

# coupon_discount
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].hist(data_all['coupon_discount'], bins=50, label='original')
ax[0].legend()
ax[1].hist(np.log1p(data_all['coupon_discount']**2), bins=50, label='log')
ax[1].legend()
plt.suptitle('coupon_discount')
plt.show()


# brand
plt.figure()
# data_all['brand'].hist(bins=50)
sns.countplot(data_all['brand'])
plt.title('brand')
plt.xticks(rotation=90)
plt.show()

data_all['brand'].nunique()    # 342 unique brands

# campaign_x
plt.figure()
sns.countplot(data_all['campaign_x'])
plt.show()

# category_code
plt.figure()
sns.countplot(data_all['category_code'])
plt.show()

# TODO: Group category_code to 3 bins (3, 7, other)

# age_range_code
plt.figure()
sns.countplot(data_all['age_range_code'])
plt.show()

# family_size_code
plt.figure()
sns.countplot(data_all['family_size_code'])
plt.show()

# no_of_children_code
plt.figure()
sns.countplot(data_all['no_of_children_code'])
plt.show()

# income_bracket_code
plt.figure()
sns.countplot(data_all['income_bracket_code'])
plt.show()

# campaign_duration
plt.figure()
data_all['campaign_duration'].hist(bins=50)
plt.show()

# trans_duration
plt.figure()
data_all['trans_duration'].hist(bins=50)
plt.show()


#%% FE Transformations
# --------------------

# selling price
data_all['selling_price_log'] = np.log1p(data_all['selling_price'])
data_all = data_all.drop(['selling_price'], axis=1)

# other_discount
data_all['other_discount_log'] = np.log1p(data_all['other_discount']**2)
data_all = data_all.drop(['other_discount'], axis=1)

# coupon_discount
data_all['coupon_discount_log'] = np.log1p(data_all['coupon_discount']**2)
data_all = data_all.drop(['coupon_discount'], axis=1)


# # export to csv
# data_all.to_csv('./data/data_encoded_3.csv', index=None)

# --------------------- OHE categ cols --------------
# OHE category_code
data_all = pd.concat([data_all, pd.get_dummies(data_all['category_code'], drop_first=True, prefix='categ_code')], axis=1)

# OHE family_size_code
data_all = pd.concat([data_all, pd.get_dummies(data_all['family_size_code'], drop_first=True, prefix='family_code')], axis=1)

# OHE age_range_code
data_all = pd.concat([data_all, pd.get_dummies(data_all['age_range_code'], drop_first=True, prefix='age_range_code')], axis=1)

# OHE income_bracket_code
data_all = pd.concat([data_all, pd.get_dummies(data_all['income_bracket_code'], drop_first=True, prefix='income_code')], axis=1)

# OHE no_of_children_code
data_all = pd.concat([data_all, pd.get_dummies(data_all['no_of_children_code'], drop_first=True, prefix='no_child_code')], axis=1)



# export to csv
data_all.to_csv('./data/data_encoded_4.csv', index=None)
# ----------------------------------------------------------------------------------------------------
