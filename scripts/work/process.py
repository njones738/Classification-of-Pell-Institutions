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

# %%
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
from sklearn.preprocessing import OneHotEncoder

# %%
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import plotnine as p9
import pandas as pd
import numpy as np

# %%
idcsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/csc_variable_subsets/id_csc.feather"
cipcsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/csc_variable_subsets/cip_csc.feather"
geoloccsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/csc_variable_subsets/geolocation_csc.feather"
instdemocsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/csc_variable_subsets/inst_demographic_csc.feather"
studdemocsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/csc_variable_subsets/stud_demographic_csc.feather"
pcipcsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/csc_variable_subsets/pcip_csc.feather"
numcsc = "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/csc_variable_subsets/num_csc.feather"

full_df_path = "data/datasubsets/csc_variable_subsets/full_csc_frame.csv"

id_csc = feather.read_feather(idcsc)
cip_csc = feather.read_feather(cipcsc)
geolocation_csc = feather.read_feather(geoloccsc)
inst_demographic_csc = feather.read_feather(instdemocsc)
stud_demographic_csc = feather.read_feather(studdemocsc)
pcip_csc = feather.read_feather(pcipcsc)
num_csc = feather.read_feather(numcsc)

id_csc
cip_csc
geolocation_csc
inst_demographic_csc
stud_demographic_csc
pcip_csc
num_csc

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

# %%
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

    print(xtrain.shape)
    print(y_train.shape)
    print(xtest.shape)
    print(y_test.shape)
    print(xvalid.shape)
    print(y_valid.shape)

    return xtrain, y_train, xtest, y_test, xvalid, y_valid 

# %%
ids = np.array(range(len(id_csc)))
id_csc["ids"] = ids

target_variable = id_csc.loc[:,["ids", "UNITID", "PELLCAT"]]
justids = id_csc.loc[:,["ids", "UNITID"]]

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
colums = ['cipPC01','cipPC02','cipPC03','cipPC04','cipPC05','cipPC06','cipPC07','cipPC08','cipPC09','cipPC10','cipPC11','cipPC12',"cipPC13","cipPC14","cipPC15","cipPC16","cipPC17","cipPC18","cipPC19","cipPC20","cipPC21","cipPC22","cipPC23","cipPC24","cipPC25","cipPC26","cipPC27","cipPC28","cipPC29","cipPC30","cipPC31","cipPC32","cipPC33","cipPC34","cipPC35","cipPC36","cipPC37"]

pca = PCA(n_components = 37)
pc = pca.fit_transform(cip_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1)) 
pca_cip = pd.DataFrame(data=pc, columns=colums) 

df = pd.DataFrame({'var':pca.explained_variance_ratio_})
comp = pca.fit_transform(cip_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1)) 
pca_cip = pd.DataFrame(comp, columns=colums)

pca_cip["ids"] = np.array(range(len(pca_cip)))

full_PCA = pd.merge(cip_csc[["ids", "UNITID", "INSTNM"]], pca_cip, how = "left",
         left_on="ids", right_on="ids")
# pcip_csc["ids"] = np.array(range(len(pcip_csc)))
colums = ['pcipPC01','pcipPC02','pcipPC03','pcipPC04','pcipPC05','pcipPC06']

pca = PCA(n_components = 6)
pc = pca.fit_transform(pcip_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1)) 
pca_pcip = pd.DataFrame(data=pc, columns=colums) # 

df = pd.DataFrame({'var':pca.explained_variance_ratio_})
comp = pca.fit_transform(pcip_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1)) 
pca_pcip = pd.DataFrame(comp, columns=colums) # , columns=colums

pca_pcip["ids"] = np.array(range(len(pca_pcip)))

full_PCA = pd.merge(full_PCA, pca_pcip, how = "left",
         left_on="ids", right_on="ids")
# geolocation_csc["ids"] = np.array(range(len(geolocation_csc)))
colums = ['geoPC01','geoPC02','geoPC03','geoPC04','geoPC05','geoPC06']

pca = PCA(n_components = 6)
pc = pca.fit_transform(geolocation_csc.drop(["ids", "UNITID", "INSTNM", "LATITUDE", "LONGITUDE", "CITY", "STABBR", "REGION", "ZIP"], axis = 1)) 
pca_geo = pd.DataFrame(data=pc, columns=colums) # 

df = pd.DataFrame({'var':pca.explained_variance_ratio_})
comp = pca.fit_transform(geolocation_csc.drop(["ids", "UNITID", "INSTNM", "LATITUDE", "LONGITUDE", "CITY", "STABBR", "ZIP"], axis = 1)) 
pca_geo = pd.DataFrame(comp, columns=colums) # , columns=colums

pca_geo["ids"] = np.array(range(len(pca_geo)))

geolocation_csc = geolocation_csc[["ids", "UNITID", "INSTNM", "LATITUDE", "LONGITUDE", "CITY", "ST_FIPS", "STABBR", "ZIP"]]
full_PCA = pd.merge(full_PCA, pca_geo, how = "left",
         left_on="ids", right_on="ids")

# %%
num_csc.drop(columns = ["ids", "UNITID", "INSTNM"], axis = 1).head(5)

num_vars = num_csc.drop(columns = ["ids", "UNITID", "INSTNM"], axis = 1)

for x in list(num_vars):
    med = num_csc[x].median()
    num_csc[x].fillna(med, inplace = True)

missing = pd.DataFrame(num_csc.isna().sum())
missing.reset_index(inplace=True)
print(missing[missing[0] == 0].shape)
missing[missing[0] != 0]

# %%
df1 = num_csc[["NUMBRANCH", "PCTFLOAN", "SCUGFFN",
         "TUITFTE", "INEXPFTE", "PPTUG_EF", "FTFTPCTFLOAN",
         "PFTFTUG1_EF", "AVGFACSAL", "NUM4_PRIV", "NPT4_PRIV",
         "NUM41_PRIV", "NUM42_PRIV", "NUM43_PRIV", "NUM44_PRIV",
         "NUM45_PRIV", "NPT4_048_PRIV", "NPT41_PRIV",
         "TUITIONFEE_IN", "TUITIONFEE_OUT", "UGDS", 
         "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", 
         "UGDS_ASIAN", "UGDS_AIAN", "UGDS_NHPI",
         "UGDS_2MOR", "UGDS_NRA", "UGDS_UNKN",
         "UGDS_MEN", "UGDS_WOMEN", "UG12MN"]]
df2 = num_csc[["OMACHT6_FTFT", "OMACHT8_FTFT", "OMACHT6_PTFT",
         "OMACHT6_FTNFT", "OMACHT8_FTNFT", "OMACHT6_PTNFT",
         "OMENRYP_FULLTIME", "OMENRAP_FULLTIME", "OMAWDP8_FULLTIME",
         "OMENRUP_FULLTIME", "OMENRYP_FIRSTTIME", "OMENRAP_FIRSTTIME",
         "OMAWDP8_FIRSTTIME", "OMENRUP_FIRSTTIME", "OMENRYP_NOTFIRSTTIME",
         "OMENRAP_NOTFIRSTTIME", "OMAWDP8_NOTFIRSTTIME", "OMENRUP_NOTFIRSTTIME",
         "OMAWDP6_FTFT", "OMAWDP8_FTFT", "OMENRYP8_FTFT", "OMENRAP8_FTFT",
         "OMENRUP8_FTFT", "OMAWDP6_FTNFT", "OMAWDP8_FTNFT", "OMENRYP8_FTNFT",
         "OMENRAP8_FTNFT", "OMENRUP8_FTNFT"]]


df5 = num_csc[["LO_INC_DEBT_MDN", "PELL_DEBT_N", "NOPELL_DEBT_N", "WDRAW_DEBT_MDN",
         "FIRSTGEN_DEBT_N", "NOTFIRSTGEN_DEBT_N", "DEP_DEBT_MDN", "IND_DEBT_MDN",
         "GRAD_DEBT_N", "WDRAW_DEBT_N", "DEP_DEBT_N", "IND_DEBT_N", "GRAD_DEBT_MDN",
         "GRAD_DEBT_MDN10YR", "LO_INC_DEBT_N", "DEBT_N", "DEBT_MDN", "FEMALE_DEBT_N",
         "MALE_DEBT_N", "MD_INC_DEBT_N", "HI_INC_DEBT_N", "PELL_DEBT_MDN",
         "NOPELL_DEBT_MDN", "FIRSTGEN_DEBT_MDN", "NOTFIRSTGEN_DEBT_MDN",
         "FEMALE_DEBT_MDN", "MALE_DEBT_MDN", "MD_INC_DEBT_MDN", "HI_INC_DEBT_MDN",
         "PPLUS_PCT_LOW", "PPLUS_PCT_HIGH", "PLUS_DEBT_INST_N", "PLUS_DEBT_INST_MD",
         "PLUS_DEBT_INST_N", "PLUS_DEBT_INST_MD"]]

num_csc = num_csc[["ids", "UNITID", "INSTNM", "BBRR2_FED_UG_N",
               "BBRR2_FED_UG_FBR","BBRR2_FED_UG_NOPROG","BBRR2_FED_UG_MAKEPROG",
               "BBRR2_FED_UGCOMP_N","BBRR2_FED_UG_DFR",
               "BBRR2_FED_UG_DFLT","BBRR2_FED_UGCOMP_NOPROG",
               "BBRR2_FED_UGCOMP_FBR","BBRR2_PP_UG_N","BBRR2_FED_UGNOCOMP_N",
               "DBRR1_FED_UG_N","DBRR1_FED_UG_NUM","DBRR1_FED_UG_DEN","DBRR1_FED_UG_RT",
               "DBRR4_FED_UG_N","DBRR4_FED_UG_NUM","DBRR4_FED_UG_DEN",
               "DBRR4_FED_UG_RT","DBRR5_FED_UG_N","DBRR5_FED_UG_NUM",
               "DBRR5_FED_UG_DEN","DBRR5_FED_UG_RT","DBRR4_FED_UGCOMP_RT",
               "DBRR4_FED_UGCOMP_N","DBRR4_FED_UGCOMP_NUM","DBRR4_FED_UGCOMP_DEN",
               "DBRR1_FED_UGCOMP_N","DBRR1_FED_UGCOMP_NUM","DBRR1_FED_UGCOMP_DEN",
               "DBRR1_FED_UGCOMP_RT","DBRR10_FED_UG_N","DBRR20_FED_UG_DEN",
               "DBRR10_FED_UG_NUM","DBRR10_FED_UG_DEN","DBRR10_FED_UG_RT",
               "DBRR4_FED_UGUNK_N","DBRR4_FED_UGUNK_NUM","DBRR4_FED_UGUNK_DEN",
               "DBRR4_FED_UGUNK_RT","DBRR4_FED_UGNOCOMP_N","DBRR4_FED_UGNOCOMP_NUM",
               "DBRR4_FED_UGNOCOMP_DEN","DBRR4_FED_UGNOCOMP_RT",
               "DBRR20_FED_UG_RT","DBRR20_FED_UG_N","DBRR20_FED_UG_NUM",
               "DBRR1_PP_UG_N","DBRR1_PP_UG_NUM","DBRR1_PP_UG_DEN","DBRR1_PP_UG_RT",
               "DBRR5_PP_UG_N","DBRR5_PP_UG_NUM","DBRR5_PP_UG_DEN","DBRR5_PP_UG_RT",
               "DBRR4_PP_UG_DEN","DBRR4_PP_UG_RT","DBRR4_PP_UG_N","DBRR4_PP_UG_NUM",
               "LPSTAFFORD_CNT", "LPSTAFFORD_AMT","LPPPLUS_CNT","LPPPLUS_AMT"]]




# %%
import seaborn as sns
%matplotlib inline

colums = ['num1PC01','num1PC02','num1PC03','num1PC04']

pca = PCA(n_components = 4)
pc = pca.fit_transform(df1) 
pca_num = pd.DataFrame(data=pc, columns=colums) # 

df = pd.DataFrame({'var':pca.explained_variance_ratio_})
comp = pca.fit_transform(df1)
pca_num = pd.DataFrame(comp, columns=colums) # , columns=colums

pca_num["ids"] = np.array(range(len(pca_num)))

full_PCA = pd.merge(full_PCA, pca_num, how = "left",
         left_on="ids", right_on="ids")
# sns.scatterplot(x=range(1, pca_num.shape[1]), y="var", data=df, color="c");
# print("SUM of first 4 components", pca.explained_variance_ratio_[0:4].sum())

colums = ['num2PC01','num2PC02']

pca = PCA(n_components = 2)
pc = pca.fit_transform(df2) 
pca_num = pd.DataFrame(data=pc, columns=colums) # 

df = pd.DataFrame({'var':pca.explained_variance_ratio_})
comp = pca.fit_transform(df2)
pca_num = pd.DataFrame(comp, columns=colums) # , columns=colums

pca_num["ids"] = np.array(range(len(pca_num)))

full_PCA = pd.merge(full_PCA, pca_num, how = "left",
         left_on="ids", right_on="ids")
# sns.scatterplot(x=range(1, pca_num.shape[1]), y="var", data=df, color="c");
# print("SUM of first 2 components", pca.explained_variance_ratio_[0:2].sum())

colums = ['num5PC01','num5PC02']

pca = PCA(n_components = 2)
pc = pca.fit_transform(df5) 
pca_num = pd.DataFrame(data=pc, columns=colums) # 

df = pd.DataFrame({'var':pca.explained_variance_ratio_})
comp = pca.fit_transform(df5)
pca_num = pd.DataFrame(comp, columns=colums) # , columns=colums

pca_num["ids"] = np.array(range(len(pca_num)))

full_PCA = pd.merge(full_PCA, pca_num, how = "left",
         left_on="ids", right_on="ids")
# sns.scatterplot(x=range(1, pca_num.shape[1]), y="var", data=df, color="c");
# print("SUM of first 2 components", pca.explained_variance_ratio_[0:2].sum())

# %%
colums = ['num5PC01','num5PC02']

pca = PCA(n_components = 10)
pc = pca.fit_transform(num_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1)) 

pca_num = pd.DataFrame(data=pc)#, columns=colums) # 

df = pd.DataFrame({'var':pca.explained_variance_ratio_})
comp = pca.fit_transform(num_csc.drop(["ids", "UNITID", "INSTNM"], axis = 1))
pca_num = pd.DataFrame(comp)#, columns=colums) # , columns=colums

pca_num["ids"] = np.array(range(len(pca_num)))

full_PCA = pd.merge(full_PCA, pca_num, how = "left",
         left_on="ids", right_on="ids")
sns.scatterplot(x=range(1, pca_num.shape[1]), y="var", data=df, color="c");
print("SUM of first 2 components", pca.explained_variance_ratio_[0:2].sum())

# %%
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

# var_names = cip_csc.drop(columns = ["ids", "UNITID", "INSTNM"], axis = 1)

# print("Original shape:", cip_csc.shape) # Original shape: (5879, 193)
# print("Number of variables that will be encoded:", var_names.shape) # (5879, 190)

# nber = 0
# for x in list(var_names):
#     enc = OneHotEncoder(handle_unknown="ignore")
#     enc.fit(np.array(cip_csc[x]).reshape(-1, 1))
#     num_category = len(np.array(enc.categories_).transpose())
#     nber = nber + num_category

# print("Number of variables to expect:", cip_csc.shape[1] + nber)

# for x in list(var_names):
#     enc = OneHotEncoder(handle_unknown="ignore")

#     enc.fit(np.array(cip_csc[x]).reshape(-1, 1))
#     labels = np.array(cip_csc[x])
#     encoded = enc.transform(labels.reshape(-1,1)).toarray()

#     num_category = len(np.array(enc.categories_).transpose())
#     var_name = np.array([(x + "_encoded{}").format(i) for i in range(num_category)])

#     for i in range(len(var_name)):
#         cip_csc[var_name[i]] = encoded[:, i]

# print(cip_csc.shape)

encoder = TargetEncoder()
encoder.fit(geolocation_csc["CITY"], id_csc["PELLCAT"])
geolocation_csc["CITY Encoded"] = encoder.transform(geolocation_csc["CITY"])

encoder = TargetEncoder()
encoder.fit(geolocation_csc["INSTNM"], id_csc["PELLCAT"])
geolocation_csc["INSTNM Encoded"] = encoder.transform(geolocation_csc["INSTNM"])

geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["ST_FIPS"].astype(int), prefix = "FIPS")],
                             axis = 1)

# geolocation_csc = pd.concat([geolocation_csc,
#                              pd.get_dummies(geolocation_csc["REGION"].astype(int), prefix = "REGION")],
#                              axis = 1)

# geolocation_csc = pd.concat([geolocation_csc,
#                              pd.get_dummies(geolocation_csc["LOCALE"].astype(int), prefix = "LOC")],
#                              axis = 1)
# ####
# geolocation_csc.loc[geolocation_csc["CCBASIC"] == 99, "CCBASIC"] = 99
# geolocation_csc = pd.concat([geolocation_csc,
#                              pd.get_dummies(geolocation_csc["CCBASIC"].astype(int), prefix = "CCB")],
#                              axis = 1)
# ####
# geolocation_csc.loc[geolocation_csc["CCSIZSET"] == 99, "CCSIZSET"] = 99
# geolocation_csc = pd.concat([geolocation_csc,
#                              pd.get_dummies(geolocation_csc["CCSIZSET"].astype(int), prefix = "CCSS")],
#                              axis = 1)
# ####
# geolocation_csc.loc[geolocation_csc["CCUGPROF"] == 99, "CCUGPROF"] = 99
# geolocation_csc = pd.concat([geolocation_csc,
#                              pd.get_dummies(geolocation_csc["CCUGPROF"].astype(int), prefix = "CCPROF")],
#                              axis = 1)
####
geolocation_csc["ZIP5"] = geolocation_csc["ZIP"].str.slice(0, 5)
geolocation_csc["ZIP4"] = geolocation_csc["ZIP"].str.slice(0, 4)
geolocation_csc["ZIP3"] = geolocation_csc["ZIP"].str.slice(0, 3)
geolocation_csc["ZIP2"] = geolocation_csc["ZIP"].str.slice(0, 2)

encoder = TargetEncoder()
encoder.fit(geolocation_csc["ZIP2"], id_csc["PELLCAT"])
geolocation_csc["ZIP2 Encoded"] = encoder.transform(geolocation_csc["ZIP2"])

encoder = TargetEncoder()
encoder.fit(geolocation_csc["ZIP3"], id_csc["PELLCAT"])
geolocation_csc["ZIP3 Encoded"] = encoder.transform(geolocation_csc["ZIP3"])

encoder = TargetEncoder()
encoder.fit(geolocation_csc["ZIP4"], id_csc["PELLCAT"])
geolocation_csc["ZIP4 Encoded"] = encoder.transform(geolocation_csc["ZIP4"])

encoder = TargetEncoder()
encoder.fit(geolocation_csc["ZIP5"], id_csc["PELLCAT"])
geolocation_csc["ZIP5 Encoded"] = encoder.transform(geolocation_csc["ZIP5"])

inst_demographic_csc = pd.concat((inst_demographic_csc,
                                  pd.DataFrame(np.array(list(inst_demographic_csc["T4APPROVALDATE"].str.split("/"))),
                                               columns = ["T4_month", "T4_day", "T4_year"])),
                                  axis = 1)

inst_demographic_csc.loc[:, ["INSTNM", "T4APPROVALDATE", "T4_month", "T4_day", "T4_year"]]


inst_demographic_csc["Season"] = inst_demographic_csc["T4_month"].map(month_dict)                                 

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["MAIN"].astype(int),
                                  prefix = "MAIN")],
                                  axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["HCM2"].astype(int),
                                  prefix = "HCM2")],
                                  axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["OPEFLAG"].astype(int),
                                  prefix = "OPEFLAG")],
                                  axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["PREDDEG"].astype(int),
                                  prefix = "PREDDEG")],
                                  axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["HIGHDEG"].astype(int),
                                  prefix = "HIGHDEG")],
                                  axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["ICLEVEL"].astype(int),
                                  prefix = "ICLEVEL")],
                                  axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["CONTROL"],
                                  prefix = "CONTROL")],
                                  axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["CURROPER"].astype(int),
                                  prefix = "CURROPER")],
                                  axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["OPENADMP"].astype(int),
                                  prefix = "OPENADMP")],
                                  axis = 1)

inst_demographic_csc = pd.concat([inst_demographic_csc,
                                  pd.get_dummies(inst_demographic_csc["ACCREDCODE"],
                                  prefix = "ACCRED")],
                                  axis = 1)

encoder = TargetEncoder()
encoder.fit(inst_demographic_csc["ACCREDAGENCY"], id_csc["PELLCAT"])
inst_demographic_csc["ACCREDAGENCY Encoded"] = encoder.transform(inst_demographic_csc["ACCREDAGENCY"])

encoder = TargetEncoder()
encoder.fit(inst_demographic_csc["Season"], id_csc["PELLCAT"])
inst_demographic_csc["Season Encoded"] = encoder.transform(inst_demographic_csc["Season"])

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["HSI"],
                                  prefix = "HSI")],
                                  axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["HBCU"],
                                  prefix = "HBCU")],
                                  axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["ANNHI"],
                                  prefix = "ANNHI")],
                                  axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["PBI"],
                                  prefix = "PBI")],
                                  axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["TRIBAL"],
                                  prefix = "TRIBAL")],
                                  axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["NANTI"],
                                  prefix = "NANTI")],
                                  axis = 1)

stud_demographic_csc = pd.concat([stud_demographic_csc,
                                  pd.get_dummies(stud_demographic_csc["AANAPII"],
                                  prefix = "AANAPII")],
                                  axis = 1)

# %%
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
full_df = pd.merge(geolocation_csc, inst_demographic_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

full_df = pd.merge(full_df, stud_demographic_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

full_df = pd.merge(full_df, num_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

full_df = pd.merge(full_df, full_PCA.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))
full_df.to_csv("full_df.csv")

#%%
id_csc.to_feather("postprocess/idcsc.feather")
cip_csc.to_feather("postprocess/cipcsc.feather")
geolocation_csc.to_feather("postprocess/geoloccsc.feather")
inst_demographic_csc.to_feather("postprocess/instdemocsc.feather")
stud_demographic_csc.to_feather("postprocess/studdemocsc.feather")
pcip_csc.to_feather("postprocess/pcipcsc.feather")
num_csc.to_feather("postprocess/numcsc.feather")
target_variable.to_feather("postprocess/tarcsc.feather")
full_df.to_feather("postprocess/fulldf.feather")

# %%
drop_rows = inst_demographic_csc.query("CONTROL != 1").query("CONTROL != 2").query("CONTROL != 3").ids
drop_rows
# inst_demographic_csc = inst_demographic_csc.drop(index = drop_rows)

# %%

list(full_df.columns)

# %%
full_df