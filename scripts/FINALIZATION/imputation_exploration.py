# %%
from pathlib import Path
import os
from joblib import Memory
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csc_matrix
from scipy.special import expit
from scipy import sparse
from sklearn.preprocessing import normalize
from copy import copy
from sklearn import svm
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from category_encoders import TargetEncoder
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder, normalize
import scikitplot as skplt

import pyarrow.feather as feather
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as p9
import pandas as pd
import numpy as np

idcsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/final_df/id_csc.feather"
cipcsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/final_df/cip_csc.feather"
geoloccsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/final_df/geolocation_csc.feather"
instdemocsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/final_df/inst_demographic_csc.feather"
studdemocsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/final_df/stud_demographic_csc.feather"
pcipcsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/final_df/pcip_csc.feather"
numcsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/final_df/num_csc.feather"

id_csc = feather.read_feather(idcsc)
cip_csc = feather.read_feather(cipcsc)
geolocation_csc = feather.read_feather(geoloccsc)
inst_demographic_csc = feather.read_feather(instdemocsc)
stud_demographic_csc = feather.read_feather(studdemocsc)
pcip_csc = feather.read_feather(pcipcsc)
num_csc = feather.read_feather(numcsc)


month_dict = {"4":"Spring", "8":"Summer", "12": "Winter",
              "2": "Winter", "1": "Winter", "7": "Summer",
              "6": "Summer", "3": "Summer", "5": "Spring",
              "11": "Winter", "10": "Fall", "9": "Fall"}
		
id_csc["UNITID"] = id_csc["UNITID"].astype(object)
id_csc["FEDSCHCD"] = id_csc["FEDSCHCD"].astype(object)
id_csc["OPEID"] = id_csc["OPEID"].astype(object)
id_csc["OPEID6"] = id_csc["OPEID6"].astype(object)


geolocation_csc.INSTNM = geolocation_csc["INSTNM"].replace("[\&]", "and", regex = True)
geolocation_csc.INSTNM = geolocation_csc["INSTNM"].replace("[\"]", "", regex = True)
geolocation_csc.INSTNM = geolocation_csc["INSTNM"].replace("[\-]", " ", regex = True)
geolocation_csc.INSTNM = geolocation_csc.INSTNM.str.lower()
inst_demographic_csc.ACCREDAGENCY = inst_demographic_csc.ACCREDAGENCY.str.lower()

rand_state = 5991

ids = np.array(range(len(id_csc)))
id_csc["ids"] = ids

target_variable = id_csc.loc[:,["ids", "UNITID", "PELLCAT"]]
justids = id_csc.loc[:,["ids", "UNITID"]]
#########################################################

#########################################################
print("cip_csc")
b4_cip = cip_csc.shape
print("BEFORE:", b4_cip)
cip_csc = pd.merge(justids, 
                   cip_csc,
                   how = "left")
af_cip = cip_csc.shape
print("AFTER: ", af_cip)

print(" ")
print("geolocation_csc")
b4_geolocation_csc = geolocation_csc.shape
print("BEFORE:", b4_geolocation_csc)
geolocation_csc = pd.merge(justids, 
                           geolocation_csc,
                           how = "left")
af_geolocation_csc = geolocation_csc.shape
print("AFTER: ", af_geolocation_csc)

print(" ")
print("inst_demographic_csc")
b4_inst_demographic_csc = inst_demographic_csc.shape
print("BEFORE:", b4_inst_demographic_csc)
inst_demographic_csc = pd.merge(justids, 
                   inst_demographic_csc,
                   how = "left")
af_inst_demographic_csc = inst_demographic_csc.shape
print("AFTER: ", af_inst_demographic_csc)

print(" ")
print("stud_demographic_csc")
b4_stud_demographic_csc = stud_demographic_csc.shape
print("BEFORE:", b4_stud_demographic_csc)
stud_demographic_csc = pd.merge(justids, 
                   stud_demographic_csc,
                   how = "left")
af_stud_demographic_csc = stud_demographic_csc.shape
print("AFTER: ", af_stud_demographic_csc)

print(" ")
print("pcip_csc")
b4_pcip_csc = pcip_csc.shape
print("BEFORE:", b4_pcip_csc)
pcip_csc = pd.merge(justids, 
                   pcip_csc,
                   how = "left")
af_pcip_csc = pcip_csc.shape
print("AFTER: ", af_pcip_csc)

print(" ")
print("num_csc")
b4_num_csc = num_csc.shape
print("BEFORE:", b4_num_csc)
num_csc = pd.merge(justids, 
                   num_csc,
                   how = "left")
af_num_csc = num_csc.shape
print("AFTER: ", af_num_csc)

# %%
full_PCA = cip_csc[["ids", "UNITID"]]
#########################################################
# INITIAL PCA
#########################################################
# colums = ['cipPC01','cipPC02','cipPC03','cipPC04','cipPC05','cipPC06','cipPC07','cipPC08','cipPC09','cipPC10','cipPC11','cipPC12',"cipPC13","cipPC14","cipPC15","cipPC16","cipPC17","cipPC18","cipPC19","cipPC20","cipPC21","cipPC22","cipPC23","cipPC24","cipPC25","cipPC26","cipPC27","cipPC28","cipPC29","cipPC30","cipPC31","cipPC32","cipPC33","cipPC34","cipPC35","cipPC36","cipPC37"]
# pca = PCA(n_components = 37)
# pc = pca.fit_transform(cip_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1)) 
# pca_cip = pd.DataFrame(data=pc, columns=colums) 
# df = pd.DataFrame({'var':pca.explained_variance_ratio_})
# comp = pca.fit_transform(cip_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1)) 
# pca_cip = pd.DataFrame(comp, columns=colums)
# pca_cip["ids"] = np.array(range(len(pca_cip)))
# full_PCA = pd.merge(cip_csc[["ids", "UNITID"]], pca_cip, how = "left",
#          left_on="ids", right_on="ids")
#########################################################
# frame = cip_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1)
# initial_feature_names = list(frame.columns)
# pca = PCA(n_components = 37, random_state = rand_state) #  
# principalComponents = pca.fit_transform(frame)
# most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.components_.shape[0])]
# most_important_names = [initial_feature_names[most_important[i]] for i in range(pca.components_.shape[0])]
# pc_df = pd.DataFrame(data = principalComponents,columns = most_important_names)
# df = pd.DataFrame({'var': pca.explained_variance_ratio_})
# pca_df = pd.DataFrame(pca.fit_transform(frame),columns = most_important_names)
# print(df[0:37].sum())
# sns.scatterplot(x=range(1, pca_df.shape[1]+1),y="var", data=df, color="c");
# interest_variable = pd.DataFrame([most_important_names,pca.explained_variance_ratio_]).T
# interest_variable.sort_values([1], ascending = False).head(50)
# skplt.decomposition.plot_pca_component_variance(pca,title='PCA Component Explained Variances of Program offerings',
#                                                 target_explained_variance=0.8,ax=None,figsize=None,title_fontsize='large',
#                                                 text_fontsize='medium')
#########################################################
# colums = ['pcipPC01','pcipPC02','pcipPC03','pcipPC04','pcipPC05','pcipPC06']
# pca = PCA(n_components = 6)
# pc = pca.fit_transform(pcip_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1)) 
# pca_pcip = pd.DataFrame(data=pc, columns=colums)
# df = pd.DataFrame({'var':pca.explained_variance_ratio_})
# comp = pca.fit_transform(pcip_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1)) 
# pca_pcip = pd.DataFrame(comp, columns=colums)
# pca_pcip["ids"] = np.array(range(len(pca_pcip)))
# full_PCA = pd.merge(full_PCA, pca_pcip, how = "left",
#          left_on="ids", right_on="ids")
#########################################################
# from sklearn import metrics
# from dalex._explainer.yhat import yhat_proba_default
# import dalex as dx
# import shapely as shap
# %%

%matplotlib inline

frame = pd.merge(cip_csc.drop(["UNITID", "INSTNM"], axis = 1),
                 pcip_csc.drop(["UNITID", "INSTNM"], axis = 1), how = "left",
         left_on="ids", right_on="ids").drop("ids", axis = 1)
initial_feature_names = list(frame.columns)

pca = PCA(n_components = 38,
          random_state = rand_state) #  
principalComponents = pca.fit_transform(frame)

most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.components_.shape[0])]
most_important_names = [initial_feature_names[most_important[i]] for i in range(pca.components_.shape[0])]

pc_df = pd.DataFrame(data = principalComponents,
                     columns = most_important_names)
df = pd.DataFrame({'var': pca.explained_variance_ratio_})
pca_df = pd.DataFrame(pca.fit_transform(frame),
                      columns = most_important_names)

print(df[0:38].sum())
sns.scatterplot(x=range(1, pca_df.shape[1]+1),y="var", data=df, color="c");

pca_df["ids"] = np.array(range(len(pca_df)))

interest_variable = pd.DataFrame([most_important_names,
                                  pca.explained_variance_ratio_]).T
interest_variable.sort_values([1], ascending = False) #.to_csv("C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/documents/cip_PCAevr.csv")

best_vars = np.unique(interest_variable[0])

skplt.decomposition.plot_pca_component_variance(pca,
                            title='PCA Component Explained Variances of Program offerings',
                            target_explained_variance=0.8,
                            ax=None,
                            figsize=None,
                            title_fontsize='large',
                            text_fontsize='medium')

rtn_frame = frame[best_vars]
rtn_frame["ids"] = np.array(range(len(rtn_frame)))

full_PCA = pd.merge(full_PCA, rtn_frame, how = "left",
         left_on="ids", right_on="ids")

#########################################################
# colums = ['geoPC01','geoPC02','geoPC03']
# pca = PCA(n_components = 3)
# pc = pca.fit_transform(geolocation_csc.drop(["ids", "UNITID", "INSTNM", "LATITUDE", "LONGITUDE", "CITY", "ST_FIPS", "STABBR", "ZIP"], axis = 1)) 
# pca_geo = pd.DataFrame(data=pc, columns=colums)
# df = pd.DataFrame({'var':pca.explained_variance_ratio_})
# comp = pca.fit_transform(geolocation_csc.drop(["ids", "UNITID", "INSTNM", "LATITUDE", "LONGITUDE", "CITY", "ST_FIPS", "STABBR", "ZIP"], axis = 1)) 
# pca_geo = pd.DataFrame(comp, columns=colums)
# pca_geo["ids"] = np.array(range(len(pca_geo)))
# geolocation_csc = geolocation_csc[["ids", "UNITID", "INSTNM", "LATITUDE", "LONGITUDE", "CITY", "ST_FIPS", "STABBR", "ZIP"]]
# full_PCA = pd.merge(full_PCA, pca_geo, how = "left",
#          left_on="ids", right_on="ids")
#########################################################  

#########################################################
## IMPUTATION
#########################################################
num_csc.drop(columns = ["ids", "UNITID", "INSTNM"], axis = 1).head(5)
num_vars = num_csc.drop(columns = ["ids", "UNITID", "INSTNM"], axis = 1)

for x in list(num_vars):
    med = num_csc[x].median()
    num_csc[x].fillna(med, inplace = True)

inst_demographic_csc["OPENADMP"].fillna(9, inplace=True)
inst_demographic_csc.ACCREDCODE = inst_demographic_csc.ACCREDCODE.replace("COE | WASCCS", "WASCCS", regex = True)
inst_demographic_csc.ACCREDCODE = inst_demographic_csc.ACCREDCODE.replace("WASCCS|WASCCS", "WASCCS", regex = True)
inst_demographic_csc["ACCREDCODE"] = inst_demographic_csc["ACCREDCODE"].fillna("NONE")
inst_demographic_csc["ACCREDAGENCY"] = inst_demographic_csc["ACCREDAGENCY"].fillna("NONE")
inst_demographic_csc["T4APPROVALDATE"].fillna("7/99/999", inplace=True)

stud_demographic_csc["HSI"].fillna(0, inplace=True)
stud_demographic_csc["HBCU"].fillna(0, inplace=True)
stud_demographic_csc["ANNHI"].fillna(0, inplace=True)
stud_demographic_csc["PBI"].fillna(0, inplace=True)
stud_demographic_csc["TRIBAL"].fillna(0, inplace=True)
stud_demographic_csc["NANTI"].fillna(0, inplace=True)
stud_demographic_csc["AANAPII"].fillna(0, inplace=True)
#########################################################
# POST-IMPUTATION PCA
#########################################################
scaler = StandardScaler()
poly = PolynomialFeatures(3)

frame = num_csc[['LPSTAFFORD_CNT','LPSTAFFORD_AMT','LPPPLUS_CNT', 'LPPPLUS_AMT']]  

scaler.fit(frame)
frame = pd.DataFrame(scaler.transform(frame), columns = list(frame.columns))

initial_feature_names = list(frame.columns)

pca = PCA(n_components = 2,
          random_state = rand_state) #  
principalComponents = pca.fit_transform(frame)

most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.components_.shape[0])]
most_important_names = [initial_feature_names[most_important[i]] for i in range(pca.components_.shape[0])]

pc_df = pd.DataFrame(data = principalComponents,
                     columns = most_important_names)
df = pd.DataFrame({'var': pca.explained_variance_ratio_})
pca_df = pd.DataFrame(pca.fit_transform(frame),
                      columns = most_important_names)

print(df[0:2].sum())
sns.scatterplot(x=range(1, pca_df.shape[1]+1),y="var", data=df, color="c");

pca_df["ids"] = np.array(range(len(pca_df)))

interest_variable = pd.DataFrame([most_important_names,
                                  pca.explained_variance_ratio_]).T
interest_variable.sort_values([1], ascending = False).head(50)
skplt.decomposition.plot_pca_component_variance(pca,
                            title='PCA Component Explained Variances of Program offerings',
                            target_explained_variance=0.8,
                            ax=None,
                            figsize=None,
                            title_fontsize='large',
                            text_fontsize='medium')
# full_PCA = pd.merge(full_PCA, pca_df, how = "left",
#          left_on="ids", right_on="ids")
# LPSTAFFORD_CNT, LPPPLUS_AMT

#########################################################
scaler = StandardScaler()
poly = PolynomialFeatures(3)

frame = num_csc[['DBRR1_FED_UG_N','DBRR1_FED_UG_NUM','DBRR1_FED_UG_DEN',
                 'DBRR1_FED_UG_RT','BBRR2_FED_UG_N','DBRR4_FED_UG_N',
                 'DBRR4_FED_UG_NUM','DBRR4_FED_UG_DEN','DBRR4_FED_UG_RT',
                 'DBRR5_FED_UG_N','DBRR5_FED_UG_NUM','DBRR5_FED_UG_DEN',
                 'DBRR5_FED_UG_RT','DBRR4_FED_UGCOMP_N', 'DBRR1_FED_UGCOMP_RT',
                 'DBRR4_FED_UGCOMP_NUM','DBRR4_FED_UGCOMP_DEN','DBRR4_FED_UGCOMP_RT',
                 'DBRR1_FED_UGCOMP_N', 'DBRR1_FED_UGCOMP_NUM','DBRR1_FED_UGCOMP_DEN',
                 'DBRR10_FED_UG_N','DBRR10_FED_UG_NUM', 'DBRR4_FED_UGNOCOMP_RT',
                 'DBRR10_FED_UG_DEN','DBRR10_FED_UG_RT','DBRR4_FED_UGUNK_N',
                 'DBRR4_FED_UGUNK_NUM','DBRR4_FED_UGUNK_DEN','DBRR4_FED_UGUNK_RT',
                 'DBRR4_FED_UGNOCOMP_N','DBRR4_FED_UGNOCOMP_NUM', 'DBRR4_FED_UGNOCOMP_DEN', 
                 'DBRR20_FED_UG_N','DBRR20_FED_UG_NUM', 'DBRR20_FED_UG_DEN', 'DBRR20_FED_UG_RT',
                 'DBRR1_PP_UG_N','DBRR1_PP_UG_NUM', 'DBRR1_PP_UG_DEN', 'DBRR1_PP_UG_RT',
                 'DBRR4_PP_UG_N', 'DBRR4_PP_UG_NUM', 'DBRR4_PP_UG_DEN', 'DBRR4_PP_UG_RT',
                 'DBRR5_PP_UG_N', 'DBRR5_PP_UG_NUM', 'DBRR5_PP_UG_DEN',
                 'DBRR5_PP_UG_RT']]

scaler.fit(frame)
frame = pd.DataFrame(scaler.transform(frame), columns = list(frame.columns))

initial_feature_names = list(frame.columns)

pca = PCA(#n_components = 5,
          random_state = rand_state) #  
principalComponents = pca.fit_transform(frame)

most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.components_.shape[0])]
most_important_names = [initial_feature_names[most_important[i]] for i in range(pca.components_.shape[0])]

pc_df = pd.DataFrame(data = principalComponents,
                     columns = most_important_names)
df = pd.DataFrame({'var': pca.explained_variance_ratio_})
pca_df = pd.DataFrame(pca.fit_transform(frame),
                      columns = most_important_names)

print(df[0:3].sum())
sns.scatterplot(x=range(1, pca_df.shape[1]+1),y="var", data=df, color="c");

pca_df["ids"] = np.array(range(len(pca_df)))

interest_variable = pd.DataFrame([most_important_names,
                                  pca.explained_variance_ratio_]).T
interest_variable.sort_values([1], ascending = False).head(10)
skplt.decomposition.plot_pca_component_variance(pca,
                            title='PCA Component Explained Variances of Loan Outcomes',
                            target_explained_variance=0.8,
                            ax=None,
                            figsize=None,
                            title_fontsize='large',
                            text_fontsize='medium')
# full_PCA = pd.merge(full_PCA, pca_df, how = "left",
#          left_on="ids", right_on="ids")
# DBRR1_FED_UGCOMP_DEN, DBRR4_FED_UGCOMP_RT, DBRR1_PP_UG_NUM
# DBRR1_FED_UGCOMP_DEN, DBRR1_PP_UG_DEN, DBRR4_FED_UG_RT

#########################################################
#########################################################
scaler = StandardScaler()
poly = PolynomialFeatures(3)

frame = num_csc[['BBRR2_FED_UG_FBR', 'BBRR2_FED_UG_NOPROG','BBRR2_FED_UG_MAKEPROG','BBRR2_FED_UGCOMP_N',
                 'BBRR2_FED_UG_DFR', 'BBRR2_FED_UG_DFLT', 'BBRR2_FED_UGCOMP_NOPROG',
                 'BBRR2_FED_UGCOMP_FBR', 'BBRR2_PP_UG_N','BBRR2_FED_UGNOCOMP_N',
               ]]

scaler.fit(frame)
frame = pd.DataFrame(scaler.transform(frame), columns = list(frame.columns))

initial_feature_names = list(frame.columns)

pca = PCA(#n_components = 5,
          random_state = rand_state) #  
principalComponents = pca.fit_transform(frame)

most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.components_.shape[0])]
most_important_names = [initial_feature_names[most_important[i]] for i in range(pca.components_.shape[0])]

pc_df = pd.DataFrame(data = principalComponents,
                     columns = most_important_names)
df = pd.DataFrame({'var': pca.explained_variance_ratio_})
pca_df = pd.DataFrame(pca.fit_transform(frame),
                      columns = most_important_names)

print(df[0:4].sum())
sns.scatterplot(x=range(1, pca_df.shape[1]+1),y="var", data=df, color="c");

pca_df["ids"] = np.array(range(len(pca_df)))

interest_variable = pd.DataFrame([most_important_names,
                                  pca.explained_variance_ratio_]).T
interest_variable.sort_values([1], ascending = False).head(10)
skplt.decomposition.plot_pca_component_variance(pca,
                            title='PCA Component Explained Variances of Loan Outcomes',
                            target_explained_variance=0.8,
                            ax=None,
                            figsize=None,
                            title_fontsize='large',
                            text_fontsize='medium')
# full_PCA = pd.merge(full_PCA, pca_df, how = "left",
#          left_on="ids", right_on="ids")
# BBRR2_FED_UG_FBR BBRR2_FED_UGCOMP_N BBRR2_FED_UG_NOPROG BBRR2_FED_UG_DFR BBRR2_FED_UG_DFLT
# BBRR2_PP_UG_N BBRR2_FED_UGCOMP_NOPROG BBRR2_FED_UG_MAKEPROG BBRR2_FED_UG_FBR BBRR2_FED_UGCOMP_N
# DBRR1_FED_UGCOMP_DEN, DBRR1_PP_UG_DEN, DBRR4_FED_UG_RT

#########################################################

scaler = StandardScaler()
poly = PolynomialFeatures(3)

frame = num_csc[['OMACHT6_FTFT','OMACHT8_FTFT','OMACHT6_PTFT','OMACHT6_FTNFT',
                 'OMACHT8_FTNFT','OMACHT6_PTNFT','OMENRYP_FULLTIME',
                 'OMENRAP_FULLTIME','OMAWDP8_FULLTIME','OMENRUP_FULLTIME',
                 'OMENRYP_FIRSTTIME','OMENRAP_FIRSTTIME','OMAWDP8_FIRSTTIME',
                 'OMENRUP_FIRSTTIME','OMENRYP_NOTFIRSTTIME','OMENRAP_NOTFIRSTTIME',
                 'OMAWDP8_NOTFIRSTTIME','OMENRUP_NOTFIRSTTIME','OMAWDP6_FTFT',
                 'OMAWDP8_FTFT','OMENRYP8_FTFT','OMENRAP8_FTFT','OMENRUP8_FTFT',
                 'OMAWDP6_FTNFT','OMAWDP8_FTNFT','OMENRYP8_FTNFT','OMENRAP8_FTNFT',
                 'OMENRUP8_FTNFT']]   


scaler.fit(frame)
frame = pd.DataFrame(scaler.transform(frame), columns = list(frame.columns))

initial_feature_names = list(frame.columns)

pca = PCA(n_components = 5,
          random_state = rand_state) #  
principalComponents = pca.fit_transform(frame)

most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.components_.shape[0])]
most_important_names = [initial_feature_names[most_important[i]] for i in range(pca.components_.shape[0])]

pc_df = pd.DataFrame(data = principalComponents,
                     columns = most_important_names)
df = pd.DataFrame({'var': pca.explained_variance_ratio_})
pca_df = pd.DataFrame(pca.fit_transform(frame),
                      columns = most_important_names)

print(df[0:5].sum())
sns.scatterplot(x=range(1, pca_df.shape[1]+1),y="var", data=df, color="c");

pca_df["ids"] = np.array(range(len(pca_df)))

interest_variable = pd.DataFrame([most_important_names,
                                  pca.explained_variance_ratio_]).T
interest_variable.sort_values([1], ascending = False).head(5)
skplt.decomposition.plot_pca_component_variance(pca,
                            title='PCA Component Explained Variances of Academic Outcomes',
                            target_explained_variance=0.8,
                            ax=None,
                            figsize=None,
                            title_fontsize='large',
                            text_fontsize='medium')
# full_PCA = pd.merge(full_PCA, pca_df, how = "left",
#          left_on="ids", right_on="ids")
# OMAWDP8_FULLTIME, OMENRAP_FULLTIME, OMACHT8_FTFT, OMENRYP_FULLTIME, OMAWDP8_FTFT
#########################################################
#########################################################

inst_demographic_csc = pd.concat((inst_demographic_csc,
                                  pd.DataFrame(np.array(list(inst_demographic_csc["T4APPROVALDATE"].str.split("/"))),
                                               columns = ["T4_month", "T4_day", "T4_year"])), axis = 1)
inst_demographic_csc["Season"] = inst_demographic_csc["T4_month"].map(month_dict)
inst_demographic_csc = inst_demographic_csc.drop("T4APPROVALDATE", axis = 1)


encoder = TargetEncoder()
encoder.fit(inst_demographic_csc["ACCREDAGENCY"], id_csc["PELLCAT"])
inst_demographic_csc["ACCREDAGENCY Encoded"] = encoder.transform(inst_demographic_csc["ACCREDAGENCY"])
inst_demographic_csc = inst_demographic_csc.drop("ACCREDAGENCY", axis = 1)

encoder = TargetEncoder()
encoder.fit(inst_demographic_csc["Season"], id_csc["PELLCAT"])
inst_demographic_csc["Season Encoded"] = encoder.transform(inst_demographic_csc["Season"])
inst_demographic_csc = inst_demographic_csc.drop(["Season"], axis = 1)

encoder = TargetEncoder()
encoder.fit(geolocation_csc["CITY"], id_csc["PELLCAT"])
geolocation_csc["CITY Encoded"] = encoder.transform(geolocation_csc["CITY"])
geolocation_csc = geolocation_csc.drop(["CITY"], axis = 1)

encoder = TargetEncoder()
encoder.fit(geolocation_csc["INSTNM"], id_csc["PELLCAT"])
geolocation_csc["INSTNM Encoded"] = encoder.transform(geolocation_csc["INSTNM"])
geolocation_csc = geolocation_csc.drop(["INSTNM", "STABBR"], axis = 1)

# geolocation_csc = pd.concat([geolocation_csc,
#                              pd.get_dummies(geolocation_csc["ST_FIPS"].astype(int), prefix = "FIPS",
#                              drop_first = True)], axis = 1).drop("ST_FIPS", axis = 1)

geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["REGION"].astype(int), prefix = "REGION",
                             drop_first = True)], axis = 1).drop("REGION", axis = 1)

geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["LOCALE"].astype(int), prefix = "LOCALE",
                             drop_first = True)], axis = 1).drop("LOCALE", axis = 1)

geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["CCBASIC"].astype(int), prefix = "CCBASIC",
                             drop_first = True)], axis = 1).drop("CCBASIC", axis = 1)

geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["CCSIZSET"].astype(int), prefix = "CCSIZSET",
                             drop_first = True)], axis = 1).drop("CCSIZSET", axis = 1)

geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["CCUGPROF"].astype(int), prefix = "CCUGPROF",
                             drop_first = True)], axis = 1).drop("CCUGPROF", axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["MAIN"].astype(int),
                                  prefix = "MAIN", drop_first = True)],
                                  axis = 1).drop("MAIN", axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["HCM2"].astype(int),
                                  prefix = "HCM2", drop_first = True)],
                                  axis = 1).drop("HCM2", axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["OPEFLAG"].astype(int),
                                  prefix = "OPEFLAG", drop_first = True)],
                                  axis = 1).drop("OPEFLAG", axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["PREDDEG"].astype(int),
                                  prefix = "PREDDEG", drop_first = True)],
                                  axis = 1).drop("PREDDEG", axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["HIGHDEG"].astype(int),
                                  prefix = "HIGHDEG", drop_first = True)],
                                  axis = 1).drop("HIGHDEG", axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["ICLEVEL"].astype(int),
                                  prefix = "ICLEVEL", drop_first = True)],
                                  axis = 1).drop("ICLEVEL", axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["CONTROL"],
                                  prefix = "CONTROL", drop_first = True)],
                                  axis = 1).drop("CONTROL", axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["CURROPER"].astype(int),
                                  prefix = "CURROPER", drop_first = True)],
                                  axis = 1).drop("CURROPER", axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["OPENADMP"].astype(int),
                                  prefix = "OPENADMP", drop_first = True)],
                                  axis = 1).drop("OPENADMP", axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["HSI"],
                                  prefix = "HSI", drop_first = True)],
                                  axis = 1).drop("HSI", axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["HBCU"],
                                  prefix = "HBCU", drop_first = True)],
                                  axis = 1).drop("HBCU", axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["ANNHI"],
                                  prefix = "ANNHI", drop_first = True)],
                                  axis = 1).drop("ANNHI", axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["PBI"],
                                  prefix = "PBI", drop_first = True)],
                                  axis = 1).drop("PBI", axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["TRIBAL"],
                                  prefix = "TRIBAL", drop_first = True)],
                                  axis = 1).drop("TRIBAL", axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["NANTI"],
                                  prefix = "NANTI", drop_first = True)],
                                  axis = 1).drop("NANTI", axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["AANAPII"],
                                  prefix = "AANAPII", drop_first = True)],
                                  axis = 1).drop("AANAPII", axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["WOMENONLY"],
                                  prefix = "WOMENONLY", drop_first = True)],
                                  axis = 1).drop("WOMENONLY", axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["MENONLY"],
                                  prefix = "MENONLY", drop_first = True)],
                                  axis = 1).drop("MENONLY", axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["DISTANCEONLY"],
                                  prefix = "DISTANCEONLY", drop_first = True)],
                                  axis = 1).drop("DISTANCEONLY", axis = 1)

print("cip_csc")
print("INITIAL BEFORE:", b4_cip)
print("INITIAL AFTER: ", af_cip)
print("POST-ENCODING: ", cip_csc.shape)
print(" ")
print("geolocation_csc")
print("INITIAL BEFORE:", b4_geolocation_csc)
print("INITIAL AFTER: ", af_geolocation_csc)
print("POST-ENCODING: ", geolocation_csc.shape)
print(" ")
print("inst_demographic_csc")
print("INITIAL BEFORE:", b4_inst_demographic_csc)
print("INITIAL AFTER: ", af_inst_demographic_csc)
print("POST-ENCODING: ", inst_demographic_csc.shape)
print(" ")
print("stud_demographic_csc")
print("INITIAL BEFORE:", b4_stud_demographic_csc)
print("INITIAL AFTER: ", af_stud_demographic_csc)
print("POST-ENCODING: ", stud_demographic_csc.shape)
print(" ")
print("pcip_csc")
print("INITIAL BEFORE:", b4_pcip_csc)
print("INITIAL AFTER: ", af_pcip_csc)
print("POST-ENCODING: ", pcip_csc.shape)
print(" ")
print("num_csc")
print("INITIAL BEFORE:", b4_num_csc)
print("INITIAL AFTER: ", af_num_csc)
print("POST-ENCODING: ", num_csc.shape)

# %%
full_df = pd.merge(geolocation_csc.drop(["ZIP", "LONGITUDE", "LATITUDE"], axis = 1), inst_demographic_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

full_df = pd.merge(full_df, stud_demographic_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

full_df = pd.merge(full_df, num_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

full_df = pd.merge(full_df, full_PCA.drop(["ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

num_lst = ["UNITID","LPSTAFFORD_CNT", "LPPPLUS_AMT", "DBRR1_FED_UGCOMP_DEN", "DBRR4_FED_UGCOMP_RT",
           "DBRR1_PP_UG_NUM", "OMAWDP8_FULLTIME", "OMENRAP_FULLTIME", "OMACHT8_FTFT", "OMENRYP_FULLTIME", 
           "OMAWDP8_FTFT"]

# num_lst = ["BBRR2_FED_UG_FBR", "BBRR2_FED_UGCOMP_N", "BBRR2_FED_UG_NOPROG", "BBRR2_FED_UG_DFR", "BBRR2_FED_UG_DFLT",
#            "BBRR2_PP_UG_N", "BBRR2_FED_UGCOMP_NOPROG", "BBRR2_FED_UG_MAKEPROG", "BBRR2_FED_UG_FBR",
#            "BBRR2_FED_UGCOMP_N", "DBRR1_FED_UGCOMP_DEN", "DBRR1_PP_UG_DEN", "DBRR4_FED_UG_RT" "OMAWDP8_FULLTIME", 
#            "OMENRAP_FULLTIME", "OMACHT8_FTFT", "OMENRYP_FULLTIME", "OMAWDP8_FTFT"]

full_df = pd.merge(full_df, num_csc[num_lst], how = 'left',
                   left_on = "UNITID", right_on = "UNITID")

full_df = full_df.drop(["D_PCTPELL_PCTFLOAN", "SCHTYPE", "ACCREDCODE"], axis = 1)

full_df["T4_month"] = full_df["T4_month"].astype(int)
full_df["T4_day"] = full_df["T4_day"].astype(int)
full_df["T4_year"] = full_df["T4_year"].astype(int)
full_df["ST_FIPS"] = full_df["ST_FIPS"].astype(int)
target_variable["PELLCAT"] = abs(1 - target_variable["PELLCAT"])

# %%
id_csc.to_feather("C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/final_df/idcsc.feather")
target_variable.to_feather("C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/final_df/tarcsc.feather")
full_df.to_feather("C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/final_df/full_df.feather")
# %%
