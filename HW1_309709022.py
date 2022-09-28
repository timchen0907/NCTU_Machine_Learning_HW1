# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 12:28:23 2022

@author: Tim Chen
"""
#%%
#################
# 載入套件&data #
#################
import pandas as pd
import numpy as np
import miceforest as mf
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
reg_data = pd.read_csv("daimonds(predict price).csv",encoding = "utf-8-sig")
reg_data = reg_data[[i for i in reg_data.columns if "Unnamed" not in i]]
reg_data
cls_data = pd.read_csv("bankruptcy(predict brankrupt or not).csv", encoding = "utf-8-sig")
cls_data = cls_data[[i for i in cls_data.columns if "Unnamed" not in i]]
cls_data

#%%
#################
# diamonds(reg) #
#################

#######################
# Fill Missing Values #
#######################

# 先找出有缺失值的欄位並檢視其欄位屬性在總資料筆數當中的占比，若大於50%將刪除該欄位
reg_missing = reg_data.isnull().sum().reset_index().rename(columns={0:'missing_num'})
reg_missing['missing_percentage']=reg_missing['missing_num']/len(reg_data)
reg_missing=reg_missing[reg_missing["missing_percentage"]>0].sort_values(by='missing_percentage',ascending=False).reset_index(drop=True)

# 由於缺失值占總筆數較少，故先將目標欄位缺失的樣本刪除，並重新統計上述資訊
reg_data = reg_data.dropna(subset = ["price"],axis = 0)
reg_missing = reg_data.isnull().sum().reset_index().rename(columns={0:'missing_num'})
reg_missing['missing_percentage']=reg_missing['missing_num']/len(reg_data)
reg_missing=reg_missing[reg_missing["missing_percentage"]>0].sort_values(by='missing_percentage',ascending=False).reset_index(drop=True)
reg_missing["col_type"] = [reg_data.dtypes[i] for i in reg_missing["index"]]

# 使用插補法進行相關補值
for i in reg_missing[reg_missing["col_type"]==object]["index"]:
    reg_data[i] = reg_data[i].astype('category')
reg_amp = mf.ampute_data(reg_data, random_state = 402)
reg_kernel = mf.ImputationKernel(reg_amp,datasets = 1, save_all_iterations = True, random_state = 402)
reg_kernel.mice(10)
print(reg_kernel)
complete_reg_data = reg_kernel.complete_data(dataset=0, inplace=False).reset_index(drop=True)

# 確定是否還有missing_values及是否填補產生新值(即不屬於原本資料類別之狀況)
reg_missing_check = complete_reg_data.isnull().sum().reset_index().rename(columns={0:'missing_num'})
reg_missing_check
np.unique(reg_data["clarity"].astype('str'))
np.unique(complete_reg_data["clarity"])


# 針對文字欄位先進行處理後，切分訓練與測試資料集
le = LabelEncoder()
for i in reg_missing[reg_missing["col_type"]==object]["index"]:
    complete_reg_data[i] = le.fit_transform(complete_reg_data[i])
X_reg = complete_reg_data.drop(columns = "price")
Y_reg = complete_reg_data["price"]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg,Y_reg, test_size = 0.3, random_state=402, shuffle=True)

#####################
# Feature Selection #
#####################

# 原始個數:9
len(X_train_reg.columns)

# 使用selectfrommodel進行挑選，由於本身特徵數不多，故threshold_使用lasso預設之1e-05挑出最不重要者即可
sfm_reg = SelectFromModel(LassoCV())
sfm_reg.fit(X_train_reg, y_train_reg)

# 最終個數:8
n_features = sfm_reg.transform(X_train_reg).shape[1]
n_features

# 將篩選apply進訓練與測試資料集
reg_new_col = X_reg.columns[sfm_reg.get_support()]
X_train_reg_fs = X_train_reg[reg_new_col]
X_test_reg_fs = X_test_reg[reg_new_col]
train_reg = pd.concat([X_train_reg_fs,y_train_reg], axis = 1).reset_index(drop=True)
test_reg = pd.concat([X_test_reg_fs, y_test_reg], axis = 1).reset_index(drop=True)

# 分別輸出excel跟csv
train_reg.to_csv("diamonds_trainingset_forhw3.csv", encoding="utf-8-sig")
test_reg.to_csv("diamonds_testingset_forhw3.csv", encoding="utf-8-sig")
train_reg.to_excel("diamonds_trainingset.xlsx")
test_reg.to_excel("diamonds_testingset.xlsx")


#%%
###################
# bankruptcy(cls) #
###################

#####################################
# Fill Missing Values with cls_data #
#####################################

# 先找出有缺失值的欄位並檢視其欄位屬性與在總資料筆數當中的占比，若大於50%將刪除該欄位
cls_missing = cls_data.isnull().sum().reset_index().rename(columns={0:'missing_num'})
cls_missing['missing_percentage']=cls_missing['missing_num']/len(cls_data)
cls_missing=cls_missing[cls_missing["missing_percentage"]>0].sort_values(by='missing_percentage',ascending=False).reset_index(drop=True)
cls_missing["col_type"] = [cls_data.dtypes[i] for i in cls_missing["index"]]

# 使用插補法進行相關補值
cls_amp = mf.ampute_data(cls_data, random_state = 402)
cls_kernel = mf.ImputationKernel(cls_amp,datasets = 1, save_all_iterations = True, random_state = 402)
cls_kernel.mice(5)
print(cls_kernel)
complete_cls_data = cls_kernel.complete_data(dataset=0, inplace=False)

# 確定是否還有missing_values及是否填補產生新值(即不屬於原本資料類別之狀況)
cls_missing_check = complete_cls_data.isnull().sum().reset_index().rename(columns={0:'missing_num'})
cls_missing_check
np.unique(cls_data[" Liability-Assets Flag"])
np.unique(complete_cls_data[" Liability-Assets Flag"])


################
# Data balance #
################

# 由於處理imbalance需在訓練資料集當中使用，測試集需保持原樣
X_cls = complete_cls_data.drop(columns="Bankrupt?")
Y_cls = complete_cls_data["Bankrupt?"]
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls,Y_cls, test_size = 0.3, random_state=402, shuffle=True)

# 訓練資料集原始label分布(0:5277筆，1:178筆)
y_train_cls.value_counts()

# 使用SMOTE
sm = SMOTEENN()
X_train_cls_smote, y_train_cls_smote = sm.fit_resample(X_train_cls, y_train_cls)

# 經SMOTE後訓練資料集label分布(0:4359筆，1:5011筆)
y_train_cls_smote.value_counts()


#####################
# Feature Selection #
#####################

# 原始個數:95
len(X_train_cls_smote.columns)

# 使用selectfrommodel進行挑選，由於本身特徵數眾多，故threshold_使用mean
sfm_cls = SelectFromModel(LogisticRegression(random_state=402, n_jobs=-1),max_features = 8,threshold="mean")
sfm_cls.fit(X_train_cls_smote, y_train_cls_smote)
sfm_cls.estimator_.coef_
# 最終個數:13
n_features = sfm_cls.transform(X_train_cls_smote).shape[1]
n_features

# 將篩選apply進訓練與測試資料集
cls_new_col = X_cls.columns[sfm_cls.get_support()]
X_train_cls_smote_fs = X_train_cls_smote[cls_new_col]
X_test_cls_fs = X_test_cls[cls_new_col]
train_cls = pd.concat([X_train_cls_smote_fs,y_train_cls_smote], axis = 1).reset_index(drop=True)
test_cls = pd.concat([X_test_cls_fs, y_test_cls], axis = 1).reset_index(drop=True)

# 分別輸出excel跟csv
train_cls.to_csv("bankruptcy_trainingset_forhw3.csv", encoding="utf-8-sig")
test_cls.to_csv("bankruptcy_testingset_forhw3.csv", encoding="utf-8-sig")
train_cls.to_excel("bankruptcy_trainingset.xlsx")
test_cls.to_excel("bankruptcy_testingset.xlsx")

