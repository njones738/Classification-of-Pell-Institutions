library(magrittr)
library(tidyverse)
library(feather)
library(tidycensus)

fip_pth <- "data/state_fp_codes.csv" # nolint
csc_pth <- "data/MERGED2018_19_PP.csv"
csc_dict_pth <- "documents/csc_dict.csv"
data_dict_pth <- "data/data_dict.csv"
data_def_pth <- "data/data_definitions.csv"

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


#######################################################################
#######################################################################

#### DATA IMPORT, CREATION of PELL CATEGORY, and ROW & COLUMN EXCLUSION

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

new_dict <- CollegeScorecard18 %>%
    see_missing() %>%
    left_join(data_dict %>%
                    select(Variable, Type, dev_cat, Definition),
              by = "Variable")
new_dict %>%
    filter(n_pct != 1) %>%
    filter(!Type %in% c("factor", "character")) %>%
    view()

new_dict %>%
    filter(n_pct != 1) %>%
    filter(Type %in% c("factor", "character")) %>%
    view()














