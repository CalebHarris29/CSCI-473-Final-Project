import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from datetime import timedelta

pd.set_option("display.max_columns", 200)
pd.set_option("max_colwidth", 200)

DATA_DIR = "data/"

ADMISSIONS_PATH = DATA_DIR + "admissions.csv"
DIAGNOSES_PATH = DATA_DIR + "diagnoses_icd.csv.gz"
PATIENTS_PATH = DATA_DIR + "patients.csv.gz"

print("Loading datasets...")

patients = pd.read_csv(PATIENTS_PATH, low_memory=False)
admissions = pd.read_csv(ADMISSIONS_PATH, low_memory=False)
diagnoses = pd.read_csv(DIAGNOSES_PATH, low_memory=False)

#helps to verify that data is loaded correctly
print("patients:", patients.shape)
print("admissions:", admissions.shape)
print("diagnoses:", diagnoses.shape)

print(admissions.head(15))

# Convert admission/discharge timestamps
admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

# Calculate length of stay in hours
admissions["length_of_stay_hours"] = (
    (admissions["dischtime"] - admissions["admittime"])
    .dt.total_seconds() / 3600
) 

print(admissions[["admittime", "dischtime", "length_of_stay_hours", "length_of_stay_hours"]].head(10))