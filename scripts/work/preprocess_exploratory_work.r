library(magrittr)
library(tidyverse)
library(feather)
library(tidycensus)

fip_pth <- "data/state_fp_codes.csv" # nolint
csc_pth <- "data/MERGED2018_19_PP.csv"
csc_dict_pth <- "documents/csc_dict.csv"
data_dict_pth <- "data/data_dict.csv"
data_def_pth <- "data/data_definitions.csv"
cen_var_path <- "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/subsubsets/cen_var.csv" # nolint
data_type_pth <- "data/full_data_dict.csv"

#######################################################################
#######################################################################

#### FUNCTIONS

# R program for getting missing values
see_missing <- function(df) {
    as_tibble(lapply(df, function(x) sum(!is.na(x)))) %>%
            pivot_longer(cols = everything(),
                         names_to = "Variable", 
                         values_to = "n"
                        ) %>%
            mutate(nmiss = max(n) - n,
                   n_pct = n / max(n),
                   nmiss_pct = nmiss / max(n))
                            }

debtoutcomes_data %>% filter(INSTNM == "Kennesaw State University") %>% see_missing() %>% arrange(desc(n)) %>% view()

ksu <- debtoutcomes_data %>% filter(INSTNM == "Kennesaw State University")

 ksu %>% 
    select(BBRR2_PP_UG_DFLT, BBRR2_PP_UG_DLNQ, BBRR2_FED_UG_DFLT, BBRR2_FED_UG_DLNQ) 
#######################################################################
#######################################################################

#### DATA IMPORT, CREATION of PELL CATEGORY, and ROW & COLUMN EXCLUSION
v19 <- load_variables(2019, "acs1", cache = TRUE)

v19 %>%
    filter(substr(name, 1, 1) == "B") %>%
    filter(!substr(name, 1, 3) %in% c("B05", "B06", "B29", "B99", "B98")) %>%
    # filter(grepl("MEDIAN INCOME| TOTAL POPULATION", concept)) %>%
    filter(grepl("", concept)) %>%
    filter(!grepl("PLACE OF BIRTH|FOREIGN-BORN|IN PUERTO RICO", concept)) %>%
    group_by(top3 = substr(name, 1, 3)) %>%
    group_by(middle3 = substr(name, 1, 6)) %>%
    group_by(top3, middle3, concept) %>%
    summarise(count = n()) %>%
    # write_csv("data/df.csv")
    view()


data_def <- read_csv(data_def_pth)
data_type <- read_csv(data_type_pth) %>%
                rename(dev_cat = "dev-cat")
data_dict <- read_csv(data_dict_pth)

desired_fips <- read_csv(fip_pth) %>%
                    filter(Mainland_plus == 1) %>% # Mainland USA + Alaska and Hawaii # nolint
                    rename(region = "Postal Code")

CollegeScorecard18 <- read_csv(csc_pth,
                               na = c("", "NA", "NaN", "NULL",
                                      "Privacy Suppressed",
                                      "PrivacySuppressed") # nolint
                              ) %>%
                      mutate(PELLCAT = case_when(PCTPELL < .5 ~ 0,  # Majority percentage of the student population recieving a Pell grant # nolint
                                                 PCTPELL >= .5 ~ 1) # Minoirty percentage of the student population recieving a Pell grant # nolint
                            )

csc_norm <- CollegeScorecard18 %>%
                select(UNITID, INSTNM, ACCREDAGENCY, ACCREDCODE,
                       ALIAS, T4APPROVALDATE, FEDSCHCD,
                       INSTURL, NPCURL)

CollegeScorecard18 %<>%
    select(-contains("POOL"), -contains("SUPP"),  # I am only using one school year, so I do not need these variables # nolint
           -INSTURL, -NPCURL, -ACCREDAGENCY, -ACCREDCODE,
           -ALIAS, -FEDSCHCD, -CIPTITLE1, -CIPTITLE2, 
           -CIPTITLE3, -CIPTITLE4, -CIPTITLE5, -CIPTITLE6,
           -CIPCODE1, -CIPCODE2, -CIPCODE3, -CIPCODE4, 
           -CIPCODE5, -CIPCODE6) %>% # I do not need the schools url's # nolint
    filter(STABBR %in% desired_fips$region) %>%
    filter(!is.na(PCTPELL))
# 5,879 observations with 2,213 variables

#######################################################################
#######################################################################

csc_id <- CollegeScorecard18 %>%
                select(UNITID, OPEID, OPEID6, INSTNM,
                       CITY, STABBR, ZIP, LONGITUDE, LATITUDE,
                       ST_FIPS, REGION, LOCALE, SCHTYPE,
                       CCBASIC, CCUGPROF, CCSIZSET,
                       CURROPER, ICLEVEL, T4APPROVALDATE,
                       OPENADMP, OPEFLAG)

missing_csc <- see_missing(CollegeScorecard18)
# missing_csc %>% view() # nolint

# missing_csc %>%
#     filter(n_pct == 1) %>%
#     view()                        # 260 variables with no missing data # nolint

# missing_csc %>%
#     filter(nmiss_pct == 1) %>%
#     view()                        # 1419 variables with all missing data # nolint

lst <- missing_csc %>%              # 799 variables with atleast some data # nolint
    filter(nmiss_pct != 1) %>%
    select(Variable)

csc <- CollegeScorecard18 %>%
            select(any_of(lst$Variable))
    
# missing_csc <- csc %>% see_missing() # nolint
# write_csv(missing_csc, "data/missing_csc.csv") # nolint
# missing_csc %>% view() # nolint

lst <- data_dict %>% filter(n_pct == 1) %>% select(Variable)

# alldata <- csc %>%
#             select(any_of(lst$Variable))

# alldata_CIP <- alldata %>%
#     select(UNITID, INSTNM, contains("CIP"), -contains("PCIP"))

# alldata_PCIP <- alldata %>%
#     select(UNITID, INSTNM, contains("PCIP"))

# alldata %<>%
#     select(-contains("CIP"), -contains("PCIP"))

# alldata
# alldata_CIP
# alldata_PCIP
# write_feather(alldata, "data/datasubsets/alldata.feather")
# write_feather(alldata_CIP, "data/datasubsets/alldata_CIP.feather")
# write_feather(alldata_PCIP, "data/datasubsets/alldata_PCIP.feather")

lst <- data_dict %>%
            filter(!Variable %in% lst$Variable)
# lst %>% view()

csc2 <- csc %>%
    select(UNITID, INSTNM, any_of(lst$Variable))
# csc2 %>%
#     slice(1:23) %>%
#     view()

satact_data <- csc2 %>%
                select(UNITID, INSTNM, contains("SAT"), contains("ACT"))

cd_data <- csc2 %>%
    select(UNITID, INSTNM,
           contains("C100"), contains("C150"), contains("C200"),
           contains("D100"), contains("D150"), contains("D200"),
           contains("RET"), contains("TRANS"))

nptnum_data <- csc2 %>%
                select(UNITID, INSTNM, contains("NPT"), contains("NUM"))

debtoutcomes_data <- csc2 %>%
                select(UNITID, INSTNM, contains("BBRR"), contains("DBRR"))

ug_data <- csc2 %>%
                select(UNITID, INSTNM, contains("UGD"), UGNONDS, GRADS,
                       UG12MN, G12MN)

cip_data <- csc2 %>%
                select(UNITID, INSTNM, contains("CIP"))

debt_data <- csc2 %>%
                select(UNITID, INSTNM, contains("MDN"), ends_with("_N"),
                       contains("MD"), LPPPLUS_CNT, LPPPLUS_AMT,
                       LPGPLUS_CNT, LPGPLUS_AMT, LPSTAFFORD_CNT, LPSTAFFORD_AMT)

studentoutcomes_data <- csc2 %>%
                select(UNITID, INSTNM, contains("OMENRUP"),
                       contains("OMACHT"), contains("OMAWDP"),
                       contains("OMENRAP"), contains("OMENRYP"))
    
lst_satact <- satact_data %>% see_missing() %>% select(Variable)
lst_cd <- cd_data %>% see_missing() %>% select(Variable)
lst_nptnum <- nptnum_data %>% see_missing() %>% select(Variable)
lst_debtoutcomes <- debtoutcomes_data %>% see_missing() %>% select(Variable)
lst_ug <- ug_data %>% see_missing() %>% select(Variable)
lst_cip <- cip_data %>% see_missing() %>% select(Variable)
lst_debt <- debt_data %>% see_missing() %>% select(Variable)
lst_studentoutcomes <- studentoutcomes_data %>% see_missing() %>% select(Variable) # nolint
lst_leftover <- data_dict %>%
            filter(Variable %in% lst$Variable) %>%
            filter(!Variable %in% lst_satact$Variable) %>%
            filter(!Variable %in% lst_cd$Variable) %>%
            filter(!Variable %in% lst_nptnum$Variable) %>%
            filter(!Variable %in% lst_debtoutcomes$Variable) %>%
            filter(!Variable %in% lst_ug$Variable) %>%
            filter(!Variable %in% lst_cip$Variable) %>%
            filter(!Variable %in% lst_debt$Variable) %>%
            filter(!Variable %in% lst_studentoutcomes$Variable)
# lst_leftover %>% view()

leftover_data <- csc %>%
                    select(UNITID, INSTNM, any_of(lst_leftover$Variable),
                           -OPEID, -OPEID6, -CITY, -STABBR, -ZIP,
                           -LONGITUDE, -LATITUDE, -ST_FIPS, -REGION,
                           -LOCALE, -SCHTYPE, -CCBASIC, -CCUGPROF, -CCSIZSET, # nolint
                           -CURROPER, -ICLEVEL, -OPENADMP, -OPEFLAG) # nolint

lst_satact
lst_cd
lst_nptnum
lst_debtoutcomes
lst_ug
lst_cip
lst_debt
lst_studentoutcomes
lst_leftover

csc_id
satact_data
cd_data
nptnum_data
debtoutcomes_data
ug_data
cip_data
debt_data
studentoutcomes_data
leftover_data

########################################################

df <- data_dict %>%
    filter(Type %in% c("factor")) %>%
    select(Variable)
CollegeScorecard18 %>%
    select(UNITID, INSTNM, any_of(df$Variable))

cscid <- see_missing(csc_id) %>% select(Variable) %>% filter(!Variable %in% c("UNITID", "INSTNM"))

csc %>%
    select(-any_of(cscid$Variable))

newlst <- csc %>%
    select(
        -contains("CIP"), -contains("BBRR"), -contains("DBRR"),
        -contains("C100"), -contains("C150"), -contains("C200"),
        -contains("D100"), -contains("D150"), -contains("D200"),
        -contains("RET"), -contains("TRANS"), -contains("NPT"),
        -contains("NUM"), -contains("OMENRUP"), -contains("OMACHT"),
        -contains("OMAWDP"), -contains("OMENRAP"), -contains("OMENRYP")
    ) %>% see_missing() %>% select(Variable)

CollegeScorecard18 %>%
    select(-contains("PLUS_DEBT_ALL")) %>%
    filter(!is.na(DEBT_MDN)) %>%
    see_missing() %>%
    filter(n_pct > 0) %>%
    view()























































