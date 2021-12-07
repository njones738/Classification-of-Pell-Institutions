import os
import time
from copy import copy
from pathlib import Path
from joblib import Memory
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import expit
from scipy.stats import pearsonr
import shap

from sklearn import svm
from sklearn import tree
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample, shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder, normalize
from sklearn import metrics
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import scikitplot as skplt

from xgboost import XGBClassifier
from category_encoders import TargetEncoder

import pyarrow.feather as feather
import matplotlib.pyplot as plt
import plotnine as p9
import pandas as pd
import numpy as np

rand_state = 5991

# Paths
idcsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/scripts/here_we_are/id_csc.feather"
tarcsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/scripts/here_we_are/tarcsc.feather"
fulldf = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/scripts/here_we_are/full_df.feather"

# FUNCTIONS
def split_data(xx, yy, testsize = 1000):
    xtrain, xtest, y_train, y_test = train_test_split(xx,
                                                      yy,
                                                      test_size = testsize,
                                                      random_state = rand_state)
    xtrain, xvalid, y_train, y_valid = train_test_split(xtrain, y_train, 
                                                        test_size = testsize,
                                                        random_state = rand_state)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    y_valid = np.array(y_valid).reshape(-1, 1)

    print(" SHAPE of xtrain:", xtrain.shape)
    print("SHAPE of y_train:", y_train.shape)
    print("  SHAPE of xtest:", xtest.shape)
    print(" SHAPE of y_test:", y_test.shape)
    print(" SHAPE of xvalid:", xvalid.shape)
    print("SHAPE of y_valid:", y_valid.shape)

    return xtrain, y_train, xtest, y_test, xvalid, y_valid 

def get_acc_auc(y, p):
    acc = np.sum(y == p) / len(y)
    auc = roc_auc_score(y, p)
    return acc, auc

def get_outs(model, train, test, valid, ytrain, ytest, yvalid):
    train_acc = round(model.score(train, ytrain), 4)
    test_acc = round(model.score(test, ytest), 4)
    valid_acc = round(model.score(valid, yvalid), 4)

    train_auc = round(roc_auc_score(ytrain, model.predict_proba(train)[:,1]), 4)
    test_auc = round(roc_auc_score(ytest, model.predict_proba(test)[:,1]), 4)
    valid_auc = round(roc_auc_score(yvalid, model.predict_proba(valid)[:,1]), 4)

    return train_auc, test_auc, valid_auc, train_acc, test_acc, valid_acc  

id_csc = feather.read_feather(idcsc)
tar_csc = feather.read_feather(tarcsc)
full_df = feather.read_feather(fulldf)

missing = pd.DataFrame(full_df.isna().sum())
missing.reset_index(inplace=True)
missing[missing[0] > 0]

ys = tar_csc["PELLCAT"].copy()
full_df["ST_FIPS"] = full_df["ST_FIPS"].astype(int)
xs = full_df.drop(["ids", "UNITID", "FTFTPCTPELL", "ST_FIPS", "LOCALE_31"], axis = 1).copy()
print("SHAPE of xs:", xs.shape)
print("SHAPE of ys:", ys.shape)
xtrain, y_train, xtest, y_test, xvalid, y_valid = split_data(xs, ys, testsize = 1000)
print(xtrain.shape)

xnot =  list(xtrain.columns[xtrain.columns.str.startswith("PELL")])
print("variables to be removed:")
print(len(xnot))
print("remaining variables:")

features = xtrain.columns
features2 = xtrain.columns.drop(xnot)

print(len(features2))

scaler = StandardScaler()
poly = PolynomialFeatures(3)

scaler.fit(xtrain[features])
xtrain2 = pd.DataFrame(scaler.transform(xtrain[features]), columns = features)
xvalid2 = pd.DataFrame(scaler.transform(xvalid[features]), columns = features)
xtest2 = pd.DataFrame(scaler.transform(xtest[features]), columns = features)

scaler.fit(xtrain[features2])
xtrain3 = pd.DataFrame(scaler.transform(xtrain[features2]), columns = features2)
xvalid3 = pd.DataFrame(scaler.transform(xvalid[features2]), columns = features2)
xtest3 = pd.DataFrame(scaler.transform(xtest[features2]), columns = features2)

print("      ORIGINAL -  xtrain before:", xtrain.shape)
print("        SCALED - xtrain2 before:", xtrain2.shape)
print("SCALED&REDUCED - xtrain3 before:", xtrain3.shape)

# SCALED AND REDUCED DATASET
corrs = []
contFeat = list(xtrain3.columns)
contFeat_length = len(contFeat)

for i in range(contFeat_length):
    for j in range(i + 1, contFeat_length):
        feati = xtrain3[contFeat[i]].values.flatten()
        featj = xtrain3[contFeat[j]].values.flatten()

        corr, _ = pearsonr(feati, featj)
        corrs.append([corr, abs(corr), contFeat[i], contFeat[j]])
correl = pd.DataFrame(corrs, columns = ["P_Corr", "P_Corr_abs", "feat1", "feat2"])
################################
grb = correl.groupby(["feat1", "feat2"]).count()
grb.sort_values("P_Corr_abs", ascending=False).to_csv("xtrain1_corrgroups.csv")
################################
tot_var_pair = (contFeat_length * (contFeat_length - 1) / 2)
num_var_g50 = len(correl[abs(correl["P_Corr"]) > 0.5])
pct_var_g50 = np.round(num_var_g50 / tot_var_pair * 100, 4)
print(xtrain3.shape)
print(int(tot_var_pair), " : Total number of features pairs:")
print(num_var_g50, "   : Number of features pairs with absolute Pearson Correl above 0.5:")
print(pct_var_g50, "% : Percent of features pairs with absolute Pearson Correl above 0.5:")

# CREDIT: Dr. Vanderheyden wrote this code.
accuracies = []
for f in features2:
    log_reg = LogisticRegression(solver = "saga",
                                 random_state = rand_state,
                                 penalty = "l1",
                                 class_weight = "balanced",
                                 max_iter = 1000)
    x = xtrain3[f].values.reshape(-1, 1)
    y = y_train.reshape(-1, 1)
    ## LIN ##############################
    log_reg.fit(x, y)
    acc, auc = get_acc_auc(y, log_reg.predict(x))
    ## LOG #############################   
    xl = np.log(x - np.min(x) + 1)
    log_reg.fit(xl, y)
    lcc, luc = get_acc_auc(y, log_reg.predict(xl))

    if lcc / acc >= 1.1 or luc / auc >= 1.05:  # if bin accuracy is 110% of linear accuracy or ... AUC is 105% ...
        xtrain3[f + "_log"] = xl
        xvalid3[f + "_log"] = np.log(xvalid3[f].values.reshape(-1, 1) - np.min(xtrain3[f])+1)
        xtest3[f + "_log"] = np.log(xtest3[f].values.reshape(-1, 1) - np.min(xtrain3[f])+1)
    ## EXP #############################   
    xe = np.exp(x)
    log_reg.fit(xe, y)
    ecc, euc = get_acc_auc(y, log_reg.predict(xe))

    if ecc / acc >= 1.1 or euc / auc >= 1.05: 
        xtrain3[f + "_exp"] = xe
        xvalid3[f + "_exp"] = np.exp(xvalid3[f].values.reshape(-1, 1))
        xtest3[f + "_exp"] = np.exp(xtest3[f].values.reshape(-1, 1))
    ## POLY ############################# 
    poly.fit(x)
    xp = poly.transform(x)
    log_reg.fit(xp, y)
    pcc, puc = get_acc_auc(y, log_reg.predict(xp))
    if pcc / acc >= 1.1 or puc / auc >= 1.05:  # if bin accuracy is 110% of linear accuracy or ... AUC is 105% ...
        xtrain3[f + "_p2"] = x**2
        xtrain3[f + "_p3"] = x**3
        xvalid3[f + "_p2"] = (xvalid3[f].values)**2
        xvalid3[f + "_p3"] = (xvalid3[f].values)**3
        xtest3[f + "_p2"] = (xtest3[f].values)**2
        xtest3[f + "_p3"] = (xtest3[f].values)**3
    ## BIN #############################
    xmin = x.min()
    rnge = x.max() - xmin

    xtrn = 0 + ((x - xmin) > 1 * rnge / 10) + ((x - xmin) > 2 * rnge / 10) + ((x - xmin) > 3 * rnge / 10) + ( # the objects in each
                (x - xmin) > 4 * rnge / 10) + ((x - xmin) > 5 * rnge / 10) + ((x - xmin) > 6 * rnge / 10) + ( # bracket returns true
                (x - xmin) > 7 * rnge / 10) + ((x - xmin) > 8 * rnge / 10) + ((x - xmin) > 9 * rnge / 10)     # or false 
    xval = 0 + ((xvalid3[f] - xmin) > 1 * rnge / 10) + ((xvalid3[f] - xmin) > 2 * rnge / 10) + ((xvalid3[f] - xmin) > 3 * rnge / 10) + (
                (xvalid3[f] - xmin) > 4 * rnge / 10) + ((xvalid3[f] - xmin) > 5 * rnge / 10) + ((xvalid3[f] - xmin) > 6 * rnge / 10) + (
                (xvalid3[f] - xmin) > 7 * rnge / 10) + ((xvalid3[f] - xmin) > 8 * rnge / 10) + ((xvalid3[f] - xmin) > 9 * rnge / 10)
    xtst = 0 + ((xtest3[f] - xmin) > 1 * rnge / 10) + ((xtest3[f] - xmin) > 2 * rnge / 10) + ((xtest3[f] - xmin) > 5 * rnge / 10) + (
                (xtest3[f] - xmin) > 3 * rnge / 10) + ((xtest3[f] - xmin) > 4 * rnge / 10) + ((xtest3[f] - xmin) > 6 * rnge / 10) + (
                (xtest3[f] - xmin) > 7 * rnge / 10) + ((xtest3[f] - xmin) > 8 * rnge / 10) + ((xtest3[f] - xmin) > 9 * rnge / 10)
                
    encoder = TargetEncoder()

    encoder.fit(xtrn, y)
    xb = encoder.transform(xtrn)
    log_reg.fit(xb, y)

    bcc, buc = get_acc_auc(y, log_reg.predict(xb))

    if bcc / acc >= 1.1 or buc / auc >= 1.05: # if bin accuracy is 110% of linear accuracy or ... AUC is 105% ...
        xtrain3[f + "_Bin"] = xb
        xvalid3[f + "_Bin"] = encoder.transform(xval)
        xtest3[f + "_Bin"] = encoder.transform(xtst)
    ## COMPLETION #############################
    lDa = lcc / acc
    eDa = ecc / acc
    pDa = pcc / acc
    bDa = bcc / acc
    lda = luc / auc
    eda = euc / auc
    pda = puc / auc
    bda = buc / auc
    accuracies.append([f, acc, lcc, ecc, pcc, bcc, auc, luc, euc, puc, buc, lDa, eDa, pDa, bDa, lda, eda, pda, bda])
###############################################

colums = ["Feature","ACC: Linear", "ACC: Log", "ACC: Exp", "ACC: Poly3","ACC: Bin",
                    "AUC: Simple Linear", "AUC: Log", "AUC: Exp","AUC: Poly3", "AUC: Bin",
                    "ACC: LOG / Linear", "ACC: EXP / Linear", "ACC: Poly3 / Linear", "ACC: Bin / Linear",
                    "AUC: LOG / Linear", "AUC: EXP / Linear", "AUC: Poly3 / Linear", "AUC: Bin / Linear"]
accDf = pd.DataFrame(accuracies, columns = colums)

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

mutual_info = mutual_info_classif(xtrain.drop('GRAD_DEBT_MDN10YR', axis = 1), y_train)
sel_five_cols = SelectKBest(mutual_info_classif, k = 10)
sel_five_cols.fit(xtrain.drop('GRAD_DEBT_MDN10YR', axis = 1), y_train)
lst = sel_five_cols.get_support()
lst = xtrain.drop('GRAD_DEBT_MDN10YR', axis = 1).columns[lst]
x_train = xtrain[lst]
x_test = xtest[lst]
x_valid = xvalid[lst]

mutual_info = mutual_info_classif(xtrain2, y_train)
sel_five_cols = SelectKBest(mutual_info_classif, k = 10)
sel_five_cols.fit(xtrain2, y_train)
lst = sel_five_cols.get_support()
lst = xtrain2.columns[lst]
x_train2 = xtrain2[lst]
x_test2 = xtest2[lst]
x_valid2 = xvalid2[lst]

ddrp_lst = [ 'GRAD_DEBT_MDN10YR','TUITIONFEE_IN_exp', 'TUITIONFEE_IN_p3', 'TUITIONFEE_OUT_p3', 'MALE_DEBT_MDN_p3', 'TUITIONFEE_OUT_p2']
mutual_info = mutual_info_classif(xtrain3.drop(ddrp_lst, axis = 1), y_train)
sel_five_cols = SelectKBest(mutual_info_classif, k = 10)
sel_five_cols.fit(xtrain3.drop(ddrp_lst, axis = 1), y_train)
lst = sel_five_cols.get_support()
lst = xtrain3.drop(ddrp_lst, axis = 1).columns[lst]
x_train3 = xtrain3[lst]
x_test3 = xtest3[lst]
x_valid3 = xvalid3[lst]

































































