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
num_csc.drop(columns = ["ids", "UNITID", "INSTNM"], axis = 1).head(5)

num_vars = num_csc.drop(columns = ["ids", "UNITID", "INSTNM"], axis = 1)

for x in list(num_vars):
    med = num_csc[x].median()
    num_csc[x].fillna(med, inplace = True)

missing = pd.DataFrame(num_csc.isna().sum())
missing.reset_index(inplace=True)
print(missing[missing[0] == 0].shape)
missing[missing[0] != 0]

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

var_names = cip_csc.drop(columns = ["ids", "UNITID", "INSTNM"], axis = 1)

print("Original shape:", cip_csc.shape) # Original shape: (5879, 193)
print("Number of variables that will be encoded:", var_names.shape) # (5879, 190)

nber = 0
for x in list(var_names):
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(np.array(cip_csc[x]).reshape(-1, 1))
    num_category = len(np.array(enc.categories_).transpose())
    nber = nber + num_category

print("Number of variables to expect:", cip_csc.shape[1] + nber)

for x in list(var_names):
    enc = OneHotEncoder(handle_unknown="ignore")

    enc.fit(np.array(cip_csc[x]).reshape(-1, 1))
    labels = np.array(cip_csc[x])
    encoded = enc.transform(labels.reshape(-1,1)).toarray()

    num_category = len(np.array(enc.categories_).transpose())
    var_name = np.array([(x + "_encoded{}").format(i) for i in range(num_category)])

    for i in range(len(var_name)):
        cip_csc[var_name[i]] = encoded[:, i]

print(cip_csc.shape)

encoder = TargetEncoder()
encoder.fit(geolocation_csc["CITY"], id_csc["PELLCAT"])
geolocation_csc["CITY Encoded"] = encoder.transform(geolocation_csc["CITY"])

encoder = TargetEncoder()
encoder.fit(geolocation_csc["INSTNM"], id_csc["PELLCAT"])
geolocation_csc["INSTNM Encoded"] = encoder.transform(geolocation_csc["INSTNM"])

geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["ST_FIPS"].astype(int), prefix = "FIPS")],
                             axis = 1)

geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["REGION"].astype(int), prefix = "REGION")],
                             axis = 1)

geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["LOCALE"].astype(int), prefix = "LOC")],
                             axis = 1)
####
geolocation_csc.loc[geolocation_csc["CCBASIC"] == 99, "CCBASIC"] = 99
geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["CCBASIC"].astype(int), prefix = "CCB")],
                             axis = 1)
####
geolocation_csc.loc[geolocation_csc["CCSIZSET"] == 99, "CCSIZSET"] = 99
geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["CCSIZSET"].astype(int), prefix = "CCSS")],
                             axis = 1)
####
geolocation_csc.loc[geolocation_csc["CCUGPROF"] == 99, "CCUGPROF"] = 99
geolocation_csc = pd.concat([geolocation_csc,
                             pd.get_dummies(geolocation_csc["CCUGPROF"].astype(int), prefix = "CCPROF")],
                             axis = 1)
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
pcip_csc["PCIP01_desc"]=pd.cut(pcip_csc["PCIP01"], bins = [-1, 0, 0.125, 0.325, 0.525, 0.725, 0.925, 1], labels = [0, 0.125, 0.325, 0.525, 0.725, 0.925, 1])
pcip_csc["PCIP03_desc"]=pd.cut(pcip_csc["PCIP03"], bins = [-1, 0, 0.007, 0.0175, 0.028, 0.0385, 0.049, 1], labels = [0, 0.007, 0.0175, 0.028, 0.0385, 0.049, 1])
pcip_csc["PCIP04_desc"]=pd.cut(pcip_csc["PCIP04"], bins = [-1, 0, 0.005, 0.0125, 0.02, 0.0275, 0.035, 1], labels = [0, 0.005, 0.0125, 0.02, 0.0275, 0.035, 1])
pcip_csc["PCIP05_desc"]=pd.cut(pcip_csc["PCIP05"], bins = [-1, 0, 0.002, 0.0035, 0.005, 0.0065, 0.008, 1], labels = [0, 0.002, 0.0035, 0.005, 0.0065, 0.008, 1])
pcip_csc["PCIP09_desc"]=pd.cut(pcip_csc["PCIP09"], bins = [-1, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 1], labels = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 1])
pcip_csc["PCIP10_desc"]=pd.cut(pcip_csc["PCIP10"], bins = [-1, 0, 0.0011, 0.0018, 0.00235, 0.0029, 0.00345, 0.004, 1], labels = [0, 0.0011, 0.0018, 0.00235, 0.0029, 0.00345, 0.004, 1])
pcip_csc["PCIP11_desc"]=pd.cut(pcip_csc["PCIP11"], bins = [-1, 0, 0.03, 0.05, 0.07, 0.09, 0.11, 1], labels = [0, 0.03, 0.05, 0.07, 0.09, 0.11, 1])
pcip_csc["PCIP12_desc"]=pd.cut(pcip_csc["PCIP12"], bins = [-1, 0, 0.0005, 0.0205, 0.0405, 0.0605, 0.0805, 1], labels = [0, 0.0005, 0.0205, 0.0405, 0.0605, 0.0805, 1])
pcip_csc["PCIP13_desc"]=pd.cut(pcip_csc["PCIP13"], bins = [-1, 0, 0.0010, 0.001625, 0.002225, 0.00285, 0.003475, 1], labels = [0, 0.0010, 0.001625, 0.002225, 0.00285, 0.003475, 1])
pcip_csc["PCIP14_desc"]=pd.cut(pcip_csc["PCIP14"], bins = [-1, 0, 0.00001, 0.03801, 0.07601, 0.11401, 0.15201, 0.19001, 1], labels = [0, 0.00001, 0.03801, 0.07601, 0.11401, 0.15201, 0.19001, 1])
pcip_csc["PCIP15_desc"]=pd.cut(pcip_csc["PCIP15"], bins = [-1, 0, 0.00001, 0.03801, 0.07601, 0.11401, 0.15201, 0.19001, 1], labels = [0, 0.00001, 0.03801, 0.07601, 0.11401, 0.15201, 0.19001, 1])
pcip_csc["PCIP16_desc"]=pd.cut(pcip_csc["PCIP16"], bins = [-1, 0, 0.001, 0.004, 0.007, 0.01, 1], labels = [0, 0.001, 0.004, 0.007, 0.01, 1])
pcip_csc["PCIP19_desc"]=pd.cut(pcip_csc["PCIP19"], bins = [-1, 0, 0.005, 0.035, 0.065, 0.095, 1], labels = [0, 0.005, 0.035, 0.065, 0.095, 1])
pcip_csc["PCIP22_desc"]=pd.cut(pcip_csc["PCIP22"], bins = [-1, 0, 0.01, 0.035, 0.065, 0.095, 1], labels = [0, 0.01, 0.035, 0.065, 0.095, 1])
pcip_csc["PCIP23_desc"]=pd.cut(pcip_csc["PCIP23"], bins = [-1, 0, 0.001, 0.012, 0.023, 0.034, 1], labels = [0, 0.001, 0.012, 0.023, 0.034, 1])
pcip_csc["PCIP24_desc"]=pd.cut(pcip_csc["PCIP24"], bins = [-1, 0, 0.001, 0.0121, 0.0241, 0.0361, 0.0481, 1], labels = [0, 0.001, 0.0121, 0.0241, 0.0361, 0.0481, 1])
pcip_csc["PCIP25_desc"]=pd.cut(pcip_csc["PCIP25"], bins = [-1, 0, 0.001, 0.0035, 0.006, 0.0085, 0.011, 0.0135, 1], labels = [0, 0.001, 0.0035, 0.006, 0.0085, 0.011, 0.0135, 1])
pcip_csc["PCIP26_desc"]=pd.cut(pcip_csc["PCIP26"], bins = [-1, 0, 0.01, 0.04, 0.07, 0.1, 0.13, 1], labels = [0, 0.01, 0.04, 0.07, 0.1, 0.13, 1])
pcip_csc["PCIP27_desc"]=pd.cut(pcip_csc["PCIP27"], bins = [-1, 0, 0.0025, 0.01, 0.0175, 0.025, 0.0325, 1], labels = [0, 0.0025, 0.01, 0.0175, 0.025, 0.0325, 1])
pcip_csc["PCIP30_desc"]=pd.cut(pcip_csc["PCIP30"], bins = [-1, 0, 0.001, 0.01, 0.019, 0.028, 0.037, 0.046, 1], labels = [0, 0.001, 0.01, 0.019, 0.028, 0.037, 0.046, 1])
pcip_csc["PCIP31_desc"]=pd.cut(pcip_csc["PCIP31"], bins = [-1, 0, 0.001, 0.01, 0.019, 0.028, 0.037, 0.046, 1], labels = [0, 0.001, 0.01, 0.019, 0.028, 0.037, 0.046, 1])
pcip_csc["PCIP38_desc"]=pd.cut(pcip_csc["PCIP38"], bins = [-1, 0, 0.001, 0.008, 0.015, 0.022, 0.029, 0.036, 1], labels = [0, 0.001, 0.008, 0.015, 0.022, 0.029, 0.036, 1])
pcip_csc["PCIP39_desc"]=pd.cut(pcip_csc["PCIP39"], bins = [-1, 0, 0.00075, 0.00625, 0.01175, 0.01725, 0.02275, 1], labels = [0, 0.00075, 0.00625, 0.01175, 0.01725, 0.02275, 1])
pcip_csc["PCIP40_desc"]=pd.cut(pcip_csc["PCIP40"], bins = [-1, 0, 0.001, 0.006, 0.011, 0.016, 0.021, 1], labels = [0, 0.001, 0.006, 0.011, 0.016, 0.021, 1])
pcip_csc["PCIP41_desc"]=pd.cut(pcip_csc["PCIP41"], bins = [-1, 0, 0.0005, 0.003, 0.0055, 0.008, 0.0105, 1], labels = [0, 0.0005, 0.003, 0.0055, 0.008, 0.0105, 1])
pcip_csc["PCIP42_desc"]=pd.cut(pcip_csc["PCIP42"], bins = [-1, 0, 0.035, 0.0475, 0.06, 0.0725, 0.085, 0.0975, 0.11, 1], labels = [0, 0.035, 0.0475, 0.06, 0.0725, 0.085, 0.0975, 0.11, 1])
pcip_csc["PCIP43_desc"]=pd.cut(pcip_csc["PCIP43"], bins = [-1, 0, 0.01, 0.03, 0.05, 0.07, 0.09, 1], labels = [0, 0.01, 0.03, 0.05, 0.07, 0.09, 1])
pcip_csc["PCIP44_desc"]=pd.cut(pcip_csc["PCIP44"], bins = [-1, 0, 0.005, 0.02, 0.035, 0.05, 0.065, 1], labels = [0, 0.005, 0.02, 0.035, 0.05, 0.065, 1])
pcip_csc["PCIP45_desc"]=pd.cut(pcip_csc["PCIP45"], bins = [-1, 0, 0.005, 0.02, 0.035, 0.05, 0.065, 1], labels = [0, 0.005, 0.02, 0.035, 0.05, 0.065, 1])
pcip_csc["PCIP46_desc"]=pd.cut(pcip_csc["PCIP46"], bins = [-1, 0, 0.0103, 0.037796, 0.125, 1], labels = [0, 0.0103, 0.037796, 0.125, 1])
pcip_csc["PCIP47_desc"]=pd.cut(pcip_csc["PCIP47"], bins = [-1, 0, 0.0005, 0.0165, 0.0315, 0.0465, 0.0615, 1], labels = [0, 0.0005, 0.0165, 0.0315, 0.0465, 0.0615, 1])
pcip_csc["PCIP48_desc"]=pd.cut(pcip_csc["PCIP48"], bins = [-1, 0, 0.0005, 0.0205, 0.0405, 0.0605, 0.0805, 0.1005, 1], labels = [0, 0.0005, 0.0205, 0.0405, 0.0605, 0.0805, 0.1005, 1])
pcip_csc["PCIP49_desc"]=pd.cut(pcip_csc["PCIP49"], bins = [-1, 0, 0.0005, 0.0205, 0.0405, 0.0605, 0.0805, 0.1005, 1], labels = [0, 0.0005, 0.0205, 0.0405, 0.0605, 0.0805, 0.1005, 1])
pcip_csc["PCIP50_desc"]=pd.cut(pcip_csc["PCIP50"], bins = [-1, 0, 0.0005, 0.0205, 0.0405, 0.0605, 0.0805, 1], labels = [0, 0.0005, 0.0205, 0.0405, 0.0605, 0.0805, 1])
pcip_csc["PCIP51_desc"]=pd.cut(pcip_csc["PCIP51"], bins = [-1, 0, 0.125, 0.325, 0.525, 0.725, 0.925, 1], labels = [0, 0.125, 0.325, 0.525, 0.725, 0.925, 1])
pcip_csc["PCIP52_desc"]=pd.cut(pcip_csc["PCIP52"], bins = [-1, 0, 0.125, 0.325, 0.525, 0.725, 0.925, 1], labels = [0, 0.125, 0.325, 0.525, 0.725, 0.925, 1])
pcip_csc["PCIP54_desc"]=pd.cut(pcip_csc["PCIP54"], bins = [-1, 0, 0.125, 0.325, 0.525, 0.725, 0.925, 1], labels = [0, 0.125, 0.325, 0.525, 0.725, 0.925, 1])

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
# cip_csc
# geolocation_csc
# inst_demographic_csc
# stud_demographic_csc
# pcip_csc
# num_csc

full_df = pd.merge(cip_csc, geolocation_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

full_df = pd.merge(full_df, inst_demographic_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

full_df = pd.merge(full_df, stud_demographic_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

full_df = pd.merge(full_df, pcip_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))

full_df = pd.merge(full_df, num_csc.drop(["INSTNM", "ids"], axis = 1), how = 'left',
                   left_on = "UNITID", right_on = "UNITID",
                   suffixes=("_cip", "_geo"))
full_df.to_csv("full_df.csv")
# %%



#%%
id_csc.to_csv("idcsc.csv")
cip_csc.to_csv("cipcsc.csv")
geolocation_csc.to_csv("geoloccsc.csv")
inst_demographic_csc.to_csv("instdemocsc.csv")
stud_demographic_csc.to_csv("studdemocsc.csv")
pcip_csc.to_csv("pcipcsc.csv")
num_csc.to_csv("numcsc.csv")
tar_csc.to_csv("tarcsc.csv")
target_variable

# %%
drop_rows = inst_demographic_csc.query("CONTROL != 1").query("CONTROL != 2").query("CONTROL != 3").ids
drop_rows
# inst_demographic_csc = inst_demographic_csc.drop(index = drop_rows)

# %%
