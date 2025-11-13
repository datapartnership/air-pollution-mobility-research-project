# Overview
# This R script analyzes panel data from Addis Ababa to examine how pollution and land use affect industrial exports. Key steps include:

# Data preprocessing: Reads raw CSV data, creates log-transformed variables, and computes interaction terms (e.g. industrial land × NO₂).

# Data cleaning: Replaces NA, NaN, Inf, and -Inf values with zero for stable model estimation.

# System GMM estimation: Uses the pgmm() function to model the dynamic relationship between exports and environmental/land-use variables, addressing endogeneity with lagged instruments.

# Model diagnostics: Applies AR(1) and AR(2) tests to check for serial correlation in residuals.

# Load libraries
library(plm)
library(lmtest)
library(sandwich)
library(lme4)
library(lmtest) 
library(readxl)
library(dplyr)
library(forecast)
library(lme4)
library(writexl)
library(car)

repo_path <- dirname(getwd())
file_path <- file.path(repo_path, "data", "temp")
setwd(file_path)
addis <- read.csv("addis-industrial-model.csv")

# Create log-transformed columns (natural log)
addis$ln_export  <- log(addis$export)
addis$ln_export_r  <- log(addis$xport.raw)
addis$ln_no2  <- log(addis$no2_mean)
addis$ln_no2_lag  <- log(addis$no2_lag1)
addis$ln_NTL  <- log(addis$NTL_mean)
addis$ln_pop  <- log(addis$pop_sum_m)
addis$ln_com  <- log(addis$lu_commercial_area)
addis$ln_ind  <- log(addis$lu_industrial_area)
addis$ln_res  <- log(addis$lu_residential_area)
addis$ln_road  <- log(addis$road_len)
addis$ln_poi  <- log(addis$poi_count)

# Create the interaction variable
addis$ind_no2 <- addis$lu_industrial_area * addis$no2_mean
addis$ln_ind_no2 <- log(addis$ind_no2)
addis$com_no2 <- addis$lu_commercial_area * addis$no2_mean
addis$ln_com_no2 <- log(addis$com_no2)

# Save to a new file
write.csv(addis, "addislog.csv", row.names = FALSE)
addislog <- read.csv("addislog.csv")

# Define the list of columns to clean
target_vars <- c("ln_export","ln_export_r","ln_no2","ln_no2_lag", "ln_NTL", "ln_pop", "ln_com", "ln_ind",
                 "ln_res", "ln_road","ln_ind_no2", "ln_poi", "ln_com_no2")

# Replace NA, NaN, -Inf, Inf with 0
addislog[target_vars] <- lapply(addislog[target_vars], function(x) {
  x[!is.finite(x)] <- 0
  return(x)
})


#System GMM Model

addis.sysgmm <- pgmm(ln_export_r ~ ln_no2 + ln_NTL + ln_poi + ln_com_no2 + ln_ind_no2  |
                       lag(ln_export_r, 4) + lag(ln_no2, 4),  # Instruments for endogenous vars
  data = addislog,
  effect = "individual",
  model = "twosteps",
  transformation = "ld",
  collapse = FALSE
  )
summary(addis.sysgmm)

mtest(addis.sysgmm, order = 1)  # AR(1)
mtest(addis.sysgmm, order = 2)  # AR(2)