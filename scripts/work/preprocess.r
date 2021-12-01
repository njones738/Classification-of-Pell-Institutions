library(magrittr)
library(tidyverse)
library(feather)
library(tidycensus)
library(sf)
library(ggplot2)

# census_api_key("d439f48f31fb83d7e880dd9cd9c4212e94fd0acb", install = TRUE)
options(tigris_use_cache = TRUE)

fip_pth <- "Classification-of-Pell-Institutions/data/datasubsets/state_fp_codes.csv" # nolint
csc_pth <- "Classification-of-Pell-Institutions/data/datasubsets/MERGED2018_19_PP.csv"
csc_dict_pth <- "Classification-of-Pell-Institutions/data/datasubsets/csc_dict.csv"
data_dict_pth <- "Classification-of-Pell-Institutions/data/datasubsets/data_dict.csv"
data_def_pth <- "Classification-of-Pell-Institutions/data/datasubsets/data_definitions.csv"
cen_var_path <- "C:/Code/GITHUB/csc/Classification-of-Pell-Institutions/data/datasubsets/subsubsets/cen_var_smaller_list.csv" # nolint
data_type_pth <- "Classification-of-Pell-Institutions/data/datasubsets/full_data_dict.csv"

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

#######################################################################
#######################################################################

#### DATA IMPORT, CREATION of PELL CATEGORY, and ROW & COLUMN EXCLUSION
desired_fips <- read_csv(fip_pth) %>%
                    filter(Mainland_plus == 1) %>% # Mainland USA + Alaska and Hawaii # nolint
                    rename(region = "Postal Code")

cen_var <- read_csv(cen_var_path) %>%
                filter(Variable != "B17022_001")

cen_data <- get_acs(geography = "tract",
                    variables = "B17022_001",
                    state = desired_fips$FIPS[1])# %>%
        #     pivot_wider(names_from = variable,
        #             values_from = c(estimate, moe))

for (i in desired_fips$FIPS) {
        temp <- get_acs(geography = "tract",
                            variables = "B17022_001",
                            state = i)# %>%
                # pivot_wider(names_from = variable,
                #             values_from = c(estimate, moe))
        cen_data <- rbind(cen_data, temp)
}
for (i in cen_var$Variable) {
        for (j in desired_fips$FIPS) {
                temp <- get_acs(geography = "tract",
                                variables = i,
                                state = j) #%>%
                        # pivot_wider(names_from = variable,
                        #         values_from = c(estimate, moe))
                cen_data <- rbind(cen_data, temp)
        }
        # cen_data <- left_join(cen_data, temp %>% select(-NAME), by = "GEOID")
}
# write_csv(cen_data, "census_data_countylvl.csv")
write_csv(cen_data, "census_data_ziplvl")

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
                            ) %>% # I do not need the schools url's # nolint
                      filter(STABBR %in% desired_fips$region) %>%
                      filter(!is.na(PCTPELL)) %>%
                      select(UNITID, OPEID, OPEID6, INSTNM, LATITUDE, LONGITUDE, # nolint
                             CITY, STABBR, ST_FIPS, REGION, ZIP, LOCALE,  # nolint
                             CCBASIC, CCSIZSET, CCUGPROF, ACCREDAGENCY, ACCREDCODE,  # nolint
                             MAIN, HCM2, OPEFLAG, T4APPROVALDATE, OPENADMP, FEDSCHCD,  # nolint
                             PREDDEG, HIGHDEG, SCHTYPE, ICLEVEL, CONTROL, CURROPER,  # nolint
                             PELLCAT, TUITFTE, TUITIONFEE_IN, TUITIONFEE_OUT, INEXPFTE,  # nolint
                             LPPPLUS_AMT, LPPPLUS_CNT, LPSTAFFORD_AMT, LPSTAFFORD_CNT,  # nolint
                             AVGFACSAL, CDR3, CDR3_DENOM, NUMBRANCH, PCTFLOAN, PCTPELL,  # nolint
                             D_PCTPELL_PCTFLOAN, FTFTPCTFLOAN, FTFTPCTPELL, SCUGFFN,  # nolint
                             PPTUG_EF, PFTFTUG1_EF, UG12MN, UGDS, UGDS_2MOR, UGDS_AIAN,  # nolint
                             UGDS_ASIAN, UGDS_BLACK, UGDS_HISP, UGDS_MEN, UGDS_NHPI,  # nolint
                             UGDS_NRA, UGDS_UNKN, UGDS_WHITE, UGDS_WOMEN,  # nolint
                             DEBT_MDN, DEBT_N, DEP_DEBT_MDN, DEP_DEBT_N,  # nolint
                             FEMALE_DEBT_MDN, FEMALE_DEBT_N, FIRSTGEN_DEBT_MDN, FIRSTGEN_DEBT_N, # nolint
                             HI_INC_DEBT_MDN, HI_INC_DEBT_N, IND_DEBT_MDN, IND_DEBT_N,  # nolint
                             LO_INC_DEBT_MDN, LO_INC_DEBT_N, MALE_DEBT_MDN, MALE_DEBT_N,  # nolint
                             MD_INC_DEBT_MDN, MD_INC_DEBT_N, NOPELL_DEBT_MDN, NOPELL_DEBT_N,  # nolint
                             NOTFIRSTGEN_DEBT_MDN, NOTFIRSTGEN_DEBT_N, PELL_DEBT_MDN, PELL_DEBT_N,  # nolint
                             WDRAW_DEBT_MDN, WDRAW_DEBT_N, GRAD_DEBT_MDN, GRAD_DEBT_N, GRAD_DEBT_MDN10YR,  # nolint
                             PLUS_DEBT_INST_MD, PLUS_DEBT_INST_N, PPLUS_PCT_HIGH, PPLUS_PCT_LOW, HSI, HBCU,  # nolint
                             ANNHI, PBI, TRIBAL, MENONLY, NANTI, WOMENONLY, AANAPII, DISTANCEONLY,  # nolint
                             OMACHT6_FTFT, OMACHT6_FTNFT, OMACHT6_PTFT, OMACHT6_PTNFT, OMAWDP6_FTFT,  # nolint
                             OMAWDP6_FTNFT, OMAWDP6_PTFT, OMAWDP6_PTNFT, OMACHT8_FTFT, OMACHT8_FTNFT,  # nolint
                             OMAWDP8_FIRSTTIME, OMENRAP_FIRSTTIME, OMENRUP_FIRSTTIME, OMENRYP_FIRSTTIME, OMAWDP8_NOTFIRSTTIME,  # nolint
                             OMENRAP_NOTFIRSTTIME, OMENRUP_NOTFIRSTTIME, OMENRYP_NOTFIRSTTIME, OMAWDP8_FTFT, OMENRAP8_FTFT,  # nolint
                             OMENRUP8_FTFT, OMENRYP8_FTFT, OMAWDP8_FTNFT, OMENRAP8_FTNFT, OMENRUP8_FTNFT, OMENRYP8_FTNFT,  # nolint
                             OMAWDP8_FULLTIME, OMENRAP_FULLTIME, OMENRUP_FULLTIME, OMENRYP_FULLTIME, DBRR1_FED_UG_DEN,  # nolint
                             DBRR1_FED_UG_N, DBRR1_FED_UG_NUM, DBRR1_FED_UG_RT, DBRR1_FED_UGCOMP_DEN, DBRR1_FED_UGCOMP_N,  # nolint
                             DBRR1_FED_UGCOMP_NUM, DBRR1_FED_UGCOMP_RT, BBRR2_FED_UG_N, BBRR2_FED_UG_DFLT, BBRR2_FED_UG_DFR,  # nolint
                             BBRR2_FED_UG_FBR, BBRR2_FED_UG_MAKEPROG, BBRR2_FED_UG_NOPROG, BBRR2_FED_UGCOMP_N, BBRR2_FED_UGCOMP_FBR,  # nolint
                             BBRR2_FED_UGCOMP_NOPROG, BBRR2_FED_UGNOCOMP_N, DBRR1_PP_UG_N, DBRR1_PP_UG_NUM, DBRR1_PP_UG_DEN,  # nolint
                             DBRR1_PP_UG_RT, BBRR2_PP_UG_N, DBRR10_FED_UG_N, DBRR10_FED_UG_DEN, DBRR10_FED_UG_NUM, DBRR10_FED_UG_RT,  # nolint
                             DBRR20_FED_UG_N, DBRR20_FED_UG_DEN, DBRR20_FED_UG_NUM, DBRR20_FED_UG_RT, DBRR4_FED_UG_N, DBRR4_FED_UG_DEN,  # nolint
                             DBRR4_FED_UG_NUM, DBRR4_FED_UG_RT, DBRR4_FED_UGCOMP_N, DBRR4_FED_UGCOMP_DEN, DBRR4_FED_UGCOMP_NUM,  # nolint
                             DBRR4_FED_UGCOMP_RT, DBRR4_FED_UGNOCOMP_N, DBRR4_FED_UGNOCOMP_DEN, DBRR4_FED_UGNOCOMP_NUM,  # nolint
                             DBRR4_FED_UGNOCOMP_RT, DBRR4_FED_UGUNK_N, DBRR4_FED_UGUNK_DEN, DBRR4_FED_UGUNK_NUM, DBRR4_FED_UGUNK_RT,  # nolint
                             DBRR4_PP_UG_N, DBRR4_PP_UG_DEN, DBRR4_PP_UG_NUM, DBRR4_PP_UG_RT, DBRR5_FED_UG_N, DBRR5_FED_UG_DEN,  # nolint
                             DBRR5_FED_UG_NUM, DBRR5_FED_UG_RT, DBRR5_PP_UG_N, DBRR5_PP_UG_DEN, DBRR5_PP_UG_NUM, DBRR5_PP_UG_RT,  # nolint
                             NPT4_048_PRIV, NPT4_PRIV, NUM4_PRIV, NPT41_PRIV, NUM41_PRIV, NUM42_PRIV, NUM43_PRIV, NUM44_PRIV,  # nolint
                             NUM45_PRIV, PCIP01, PCIP03, PCIP04, PCIP05, PCIP09, PCIP10, PCIP11, PCIP12, PCIP13, PCIP14, PCIP15,  # nolint
                             PCIP16, PCIP19, PCIP22, PCIP23, PCIP24, PCIP25, PCIP26, PCIP27, PCIP29, PCIP30, PCIP31, PCIP38, PCIP39,  # nolint
                             PCIP40, PCIP41, PCIP42, PCIP43, PCIP44, PCIP45, PCIP46, PCIP47, PCIP48, PCIP49, PCIP50, PCIP51, PCIP52,  # nolint
                             PCIP54, CIP01ASSOC, CIP01BACHL, CIP01CERT1, CIP01CERT2, CIP01CERT4, CIP03ASSOC, CIP03BACHL, CIP03CERT1,  # nolint
                             CIP03CERT2, CIP03CERT4, CIP04ASSOC, CIP04BACHL, CIP04CERT1, CIP04CERT2, CIP04CERT4, CIP05ASSOC, CIP05BACHL,  # nolint
                             CIP05CERT1, CIP05CERT2, CIP05CERT4, CIP09ASSOC, CIP09BACHL, CIP09CERT1, CIP09CERT2, CIP09CERT4, CIP10ASSOC,  # nolint
                             CIP10BACHL, CIP10CERT1, CIP10CERT2, CIP10CERT4, CIP11ASSOC, CIP11BACHL, CIP11CERT1, CIP11CERT2, CIP11CERT4,  # nolint
                             CIP12ASSOC, CIP12BACHL, CIP12CERT1, CIP12CERT2, CIP12CERT4, CIP13ASSOC, CIP13BACHL, CIP13CERT1, CIP13CERT2,  # nolint
                             CIP13CERT4, CIP14ASSOC, CIP14BACHL, CIP14CERT1, CIP14CERT2, CIP14CERT4, CIP15ASSOC, CIP15BACHL, CIP15CERT1,  # nolint
                             CIP15CERT2, CIP15CERT4, CIP16ASSOC, CIP16BACHL, CIP16CERT1, CIP16CERT2, CIP16CERT4, CIP19ASSOC, CIP19BACHL,  # nolint
                             CIP19CERT1, CIP19CERT2, CIP19CERT4, CIP22ASSOC, CIP22BACHL, CIP22CERT1, CIP22CERT2, CIP22CERT4, CIP23ASSOC,  # nolint
                             CIP23BACHL, CIP23CERT1, CIP23CERT2, CIP23CERT4, CIP24ASSOC, CIP24BACHL, CIP24CERT1, CIP24CERT2, CIP24CERT4,  # nolint
                             CIP25ASSOC, CIP25BACHL, CIP25CERT1, CIP25CERT2, CIP25CERT4, CIP26ASSOC, CIP26BACHL, CIP26CERT1, CIP26CERT2,  # nolint
                             CIP26CERT4, CIP27ASSOC, CIP27BACHL, CIP27CERT1, CIP27CERT2, CIP27CERT4, CIP29ASSOC, CIP29BACHL, CIP29CERT1,  # nolint
                             CIP29CERT2, CIP29CERT4, CIP30ASSOC, CIP30BACHL, CIP30CERT1, CIP30CERT2, CIP30CERT4, CIP31ASSOC, CIP31BACHL,  # nolint
                             CIP31CERT1, CIP31CERT2, CIP31CERT4, CIP38ASSOC, CIP38BACHL, CIP38CERT1, CIP38CERT2, CIP38CERT4, CIP39ASSOC,  # nolint
                             CIP39BACHL, CIP39CERT1, CIP39CERT2, CIP39CERT4, CIP40ASSOC, CIP40BACHL, CIP40CERT1, CIP40CERT2, CIP40CERT4,  # nolint
                             CIP41ASSOC, CIP41BACHL, CIP41CERT1, CIP41CERT2, CIP41CERT4, CIP42ASSOC, CIP42BACHL, CIP42CERT1, CIP42CERT2,  # nolint
                             CIP42CERT4, CIP43ASSOC, CIP43BACHL, CIP43CERT1, CIP43CERT2, CIP43CERT4, CIP44ASSOC, CIP44BACHL, CIP44CERT1,  # nolint
                             CIP44CERT2, CIP44CERT4, CIP45ASSOC, CIP45BACHL, CIP45CERT1, CIP45CERT2, CIP45CERT4, CIP46ASSOC, CIP46BACHL,  # nolint
                             CIP46CERT1, CIP46CERT2, CIP46CERT4, CIP47ASSOC, CIP47BACHL, CIP47CERT1, CIP47CERT2, CIP47CERT4, CIP48ASSOC,  # nolint
                             CIP48BACHL, CIP48CERT1, CIP48CERT2, CIP48CERT4, CIP49ASSOC, CIP49BACHL, CIP49CERT1, CIP49CERT2, CIP49CERT4,  # nolint
                             CIP50ASSOC, CIP50BACHL, CIP50CERT1, CIP50CERT2, CIP50CERT4, CIP51ASSOC, CIP51BACHL, CIP51CERT1, CIP51CERT2,  # nolint
                             CIP51CERT4, CIP52ASSOC, CIP52BACHL, CIP52CERT1, CIP52CERT2, CIP52CERT4, CIP54ASSOC, CIP54BACHL, CIP54CERT1,  # nolint
                             CIP54CERT2, CIP54CERT4)
#######################################################################
names(CollegeScorecard18)
CollegeScorecard18 %>%
        group_by(PELLCAT) %>%
        summarise(count = n())

CollegeScorecard18 %>%
        see_missing() %>%
        filter(n_pct != 1)
        view()

#######################################################################

#### DATA SUBSETTING

## categorical variables
id_csc <- CollegeScorecard18 %>% select(UNITID, FEDSCHCD, OPEID, OPEID6, INSTNM, PELLCAT, PCTPELL) # nolint
cip_csc <- CollegeScorecard18 %>% select(UNITID, INSTNM, contains("CIP"), -contains("PCIP")) # nolint
geolocation_csc <- CollegeScorecard18 %>% select(UNITID, INSTNM, LATITUDE, LONGITUDE, CITY, ST_FIPS, # nolint
                                                 STABBR, REGION, ZIP, LOCALE, CCBASIC, CCSIZSET, CCUGPROF) # nolint
inst_demographic_csc <- CollegeScorecard18 %>% select(UNITID, INSTNM, ACCREDAGENCY, ACCREDCODE, MAIN, HCM2, # nolint
                                                      OPEFLAG, T4APPROVALDATE, OPENADMP, PREDDEG, HIGHDEG, # nolint
                                                      SCHTYPE, ICLEVEL, CONTROL, CURROPER) # nolint
stud_demographic_csc <- CollegeScorecard18 %>% select(UNITID, INSTNM, HSI, HBCU, ANNHI, PBI, # nolint
                                                      TRIBAL, MENONLY, NANTI, WOMENONLY, # nolint
                                                      AANAPII, DISTANCEONLY) # nolint

## numeric variables
pcip_csc <- CollegeScorecard18 %>% select(UNITID, INSTNM, contains("PCIP"))

new_lst <- data_dict %>%
    filter(!Variable %in% c(names(id_csc), names(cip_csc), names(geolocation_csc), # nolint
                            names(inst_demographic_csc), names(stud_demographic_csc), # nolint
                            names(pcip_csc))) %>%
    filter(!Type %in% c("character", "factor")) %>%
    select(Variable) %>% 
    as.list()

num_csc <- CollegeScorecard18 %>% select(UNITID, INSTNM, any_of(new_lst$Variable)) # nolint

# id_csc
# cip_csc
# geolocation_csc
# inst_demographic_csc
# stud_demographic_csc
# pcip_csc
# num_csc

idcsc <- "data/datasubsets/csc_variable_subsets/id_csc.feather"
cipcsc <- "data/datasubsets/csc_variable_subsets/cip_csc.feather"
geoloccsc <- "data/datasubsets/csc_variable_subsets/geolocation_csc.feather"
instdemocsc <- "data/datasubsets/csc_variable_subsets/inst_demographic_csc.feather"
studdemocsc <- "data/datasubsets/csc_variable_subsets/stud_demographic_csc.feather"
pcipcsc <- "data/datasubsets/csc_variable_subsets/pcip_csc.feather"
numcsc <- "data/datasubsets/csc_variable_subsets/num_csc.feather"

write_feather(id_csc, idcsc)
write_feather(cip_csc, cipcsc)
write_feather(geolocation_csc, geoloccsc)
write_feather(inst_demographic_csc, instdemocsc)
write_feather(stud_demographic_csc, studdemocsc)
write_feather(pcip_csc, pcipcsc)
write_feather(num_csc, numcsc)

#######################################################################
#######################################################################

# idcsc <- "data/datasubsets/csc_variable_subsets/id_csc.feather"
# cipcsc <- "data/datasubsets/csc_variable_subsets/cip_csc.feather"
# geoloccsc <- "data/datasubsets/csc_variable_subsets/geolocation_csc.feather"
# instdemocsc <- "data/datasubsets/csc_variable_subsets/inst_demographic_csc.feather"
# studdemocsc <- "data/datasubsets/csc_variable_subsets/stud_demographic_csc.feather"
# pcipcsc <- "data/datasubsets/csc_variable_subsets/pcip_csc.feather"
# numcsc <- "data/datasubsets/csc_variable_subsets/num_csc.feather"

# idcsc <- read_feather(idcsc)
# cipcsc <- read_feather(cipcsc)
# geoloccsc <- read_feather(geoloccsc)
# instdemocsc <- read_feather(instdemocsc)
# studdemocsc <- read_feather(studdemocsc)
# pcipcsc <- read_feather(pcipcsc)
# numcsc <- read_feather(numcsc)

see_missing(idcsc) %>%
        left_join(data_dict %>% select(Variable, Type, dev_cat, Definition),
                  by = "Variable") %>%
        view(title = "idcsc")
        # write_csv("data/idcsc_names.csv")
see_missing(cipcsc) %>%
        left_join(data_dict %>% select(Variable, Type, dev_cat, Definition),
                  by = "Variable") %>%
        view(title = "cipcsc")
        # write_csv("data/cipcsc_names.csv")
see_missing(geoloccsc) %>%
        left_join(data_dict %>% select(Variable, Type, dev_cat, Definition),
                  by = "Variable") %>%
        view(title = "geoloccsc")
        # write_csv("data/geoloccsc_names.csv")
see_missing(instdemocsc) %>%
        left_join(data_dict %>% select(Variable, Type, dev_cat, Definition),
                  by = "Variable") %>%
        view(title = "instdemocsc")
        # write_csv("data/instdemocsc_names.csv")
see_missing(studdemocsc) %>%
        left_join(data_dict %>% select(Variable, Type, dev_cat, Definition),
                  by = "Variable") %>%
        view(title = "studdemocsc")
        # write_csv("data/studdemocsc_names.csv")
see_missing(pcipcsc) %>%
        left_join(data_dict %>% select(Variable, Type, dev_cat, Definition),
                  by = "Variable") %>%
        view(title = "pcipcsc")
        # write_csv("data/pcipcsc_names.csv")
see_missing(numcsc) %>%
        left_join(data_dict %>% select(Variable, Type, dev_cat, Definition),
                  by = "Variable") %>%
        view(title = "numcsc")
        # write_csv("data/numcsc_names.csv")

fdf <- read_csv("scripts/full_df.csv")

lst <- fdf %>%
        select(-contains("encoded"), -contains("desc"),
               -contains("CERT"), -contains("ASSOC"),
               -contains("BACHL"), -contains("FIPS"),
               -contains("REGION"), -contains("CCB_"),
               -contains("LOC_"), -contains("CCPROF_"),
               -contains("ZIP"), -contains("OPEFLAG"),
               -contains("HCM2"), -contains("Season"),
               -starts_with("T4"), -contains("ICLEVEL"), 
               -contains("PREDDEG"), -contains("HIGHDEG"),
               -contains("CONTROL"), -contains("CUR"),
               -contains("ACCRED"), -contains("CUR"),
               -contains("HSI"), -contains("MENONLY"),
               -contains("HBCU"), -contains("TRIBAL"),
               -contains("ANNHI"), -contains("PBI"),
               -contains("CCSS"), -contains("MAIN"),
               -contains("OPENADMP"), -contains("NANTI"),
               -contains("ccss"), -contains("AANAPII"),
               -INSTNM, -CITY, -STABBR, -LOCALE, -CCBASIC,
               -CCSIZSET, -CCUGPROF, -DISTANCEONLY, -D_PCTPELL_PCTFLOAN,
               -"...1", -"ids", "CITY Encoded", "INSTNM Encoded",
               "ZIP2 Encoded", "ZIP3 Encoded", "ZIP4 Encoded", "ZIP5 Encoded",
               "Season Encoded", contains("BALANCE_REMAINING")
               ) %>%
        # write_csv("numeric_only.csv")
        # see_missing() %>%
        # select(Variable) %>%
        # as.list()
        view()

fdf %>%
        select(-any_of(lst$Variable)) %>%
        # see_missing() %>%
        write_csv("categorical_only.csv")
