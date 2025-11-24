import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from datetime import timedelta

pd.set_option("display.max_columns", 200)
pd.set_option("max_colwidth", 200)

DATA_DIR = "data/"

ADMISSIONS_PATH = DATA_DIR + "admissions.csv.gz"
DIAGNOSES_PATH = DATA_DIR + "diagnoses_icd.csv.gz"
PATIENTS_PATH = DATA_DIR + "patients.csv.gz"

print("Loading datasets...")

patients = pd.read_csv(PATIENTS_PATH, low_memory=False)
admissions = pd.read_csv(ADMISSIONS_PATH, low_memory=False)
diagnoses = pd.read_csv(DIAGNOSES_PATH, low_memory=False)

#helps to verify that data is loaded correctly
# print("patients:", patients.shape)
# print("admissions:", admissions.shape)
# print("diagnoses:", diagnoses.shape)

# print(admissions.head(15))

# Convert admission/discharge timestamps
admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

# Calculate length of stay in hours
admissions["length_of_stay_hours"] = (
    (admissions["dischtime"] - admissions["admittime"])
    .dt.total_seconds() / 3600
) 

# print(admissions[["admittime", "dischtime", "length_of_stay_hours", "length_of_stay_hours"]].head(10))

admissions = admissions.sort_values(["subject_id", "admittime"])

#get the next admission time
admissions["next_admittime"] = admissions.groupby("subject_id")["admittime"].shift(-1)

#time between discharge and next admission
admissions["days_until_next_admit"] = (
    (admissions["next_admittime"] - admissions["dischtime"])
    .dt.total_seconds() / 86400    # convert seconds to days
)

#    1 = readmitted within 30 days
#    0 = not readmitted

admissions["readmitted_within_30d"] = (
    (admissions["days_until_next_admit"] >= 0) &
    (admissions["days_until_next_admit"] <= 30)
).astype(int)

# Preview results to confirm it worked
# print(admissions[[
#     "subject_id", "hadm_id", "admittime", "dischtime",
#     "next_admittime", "days_until_next_admit",
#     "readmitted_within_30d"
# ]].head(15))

"""Feature engineering - create age, length_of_stay, prior_admit_count, comorbidity_count"""

# Merge datasets - admissions with patient demographics
merged_df = admissions.merge(patients, on="subject_id", how="left")

# Calculate age at admission
merged_df["age"] = merged_df["anchor_age"]
# Convert admission/discharge timestamps
merged_df["length_of_stay_days"] = merged_df["length_of_stay_hours"] / 24

merged_df = merged_df.sort_values(["subject_id", "admittime"])
# Count prior admissions
merged_df["prior_admit_count"] = merged_df.groupby("subject_id").cumcount()
# Calculate comorbidity count - number of unique ICD codes per patient
comorb = diagnoses.groupby("subject_id")["icd_code"].nunique().reset_index()
comorb.columns = ["subject_id", "comorbidity_count"]
merged_df = merged_df.merge(comorb, on="subject_id", how="left")

"""Handle missing values - fill numerical with median, categorical with mode"""
# Identify numerical and categorical columns
num_cols = ["age","length_of_stay_days","prior_admit_count","comorbidity_count"]
cat_cols = ["gender","race","discharge_location"]
# Fill missing values
merged_df[num_cols] = merged_df[num_cols].fillna(merged_df[num_cols].median())
merged_df[cat_cols] = merged_df[cat_cols].fillna(merged_df[cat_cols].mode().iloc[0])

"""Group specific categories into broader groups"""
def collapse_race(r):
    r = str(r).upper()
    if "ASIAN" in r:
        return "ASIAN"
    elif "BLACK" in r:
        return "BLACK"
    elif "HISPANIC" in r or "LATINO" in r:
        return "HISPANIC"
    elif "WHITE" in r:
        return "WHITE"
    elif "NATIVE HAWAIIAN" in r or "PACIFIC" in r:
        return "NATIVE_HAWAIIAN_PACIFIC"
    else:
        return "OTHER"

merged_df["race_broad"] = merged_df["race"].apply(collapse_race)

#drop original race column from the dataset
merged_df = merged_df.drop(columns=["race"])
# separate categorical variables into binary dummies
merged_df = pd.get_dummies(
    merged_df,
    columns=["gender", "discharge_location", "race_broad"],
    drop_first=True
)


print(merged_df.columns)
print(merged_df.head())
merged_df.isna().sum()

import os
os.makedirs("output", exist_ok=True)

output_path = "output/cleaned_dataset.csv"
merged_df.to_csv(output_path, index=False)

print(f"\nSaved cleaned dataset to: {output_path}")
