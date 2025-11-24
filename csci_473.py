
"""CSCI 473.ipynb"""

from google.colab import files
uploaded = files.upload()

import pandas as pd

patients = pd.read_csv("patients.csv")
admissions = pd.read_csv("admissions.csv")

patients.head()
admissions.head()

"""
Readmission Prediction Script
- Cleans demographic data (race, age, gender)
- Creates 30-day readmission label
- Builds feature matrix
- Trains Logistic Regression and Random Forest
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

# ==========================================================
# 1. LOAD DATA  (make sure patients.csv and admissions.csv
#    are uploaded into the Colab environment)
# ==========================================================

patients_path = "patients.csv"
admissions_path = "admissions.csv"

patients = pd.read_csv(patients_path)
admissions = pd.read_csv(admissions_path)

print("Patients columns:", patients.columns.tolist())
print("Admissions columns:", admissions.columns.tolist())

# ==========================================================
# 2. CONFIG: COLUMN NAMES (EDIT HERE IF YOUR NAMES DIFFER)
# ==========================================================

SUBJECT_COL = "subject_id"

# Age column (from patients.csv)
AGE_COL = None
for cand in ["age", "anchor_age", "Age", "AGE"]:
    if cand in patients.columns:
        AGE_COL = cand
        break
if AGE_COL is None:
    raise ValueError("Could not find an age column in patients.csv")

# Gender column (from patients.csv)
GENDER_COL = None
for cand in ["gender", "Gender", "sex", "Sex"]:
    if cand in patients.columns:
        GENDER_COL = cand
        break
if GENDER_COL is None:
    raise ValueError("Could not find a gender column in patients.csv")

# Admission & discharge time columns (from admissions.csv)
ADMIT_COL = None
for cand in ["admittime", "admit_time", "admission_time", "admission_datetime"]:
    if cand in admissions.columns:
        ADMIT_COL = cand
        break
if ADMIT_COL is None:
    raise ValueError("Could not find an admission time column in admissions.csv")

DISCH_COL = None
for cand in ["dischtime", "discharge_time", "discharge_datetime"]:
    if cand in admissions.columns:
        DISCH_COL = cand
        break
if DISCH_COL is None:
    raise ValueError("Could not find a discharge time column in admissions.csv")

# Discharge disposition / location (from admissions.csv)
DISP_COL = None
for cand in ["discharge_disposition", "disposition", "discharge_location"]:
    if cand in admissions.columns:
        DISP_COL = cand
        break
if DISP_COL is None:
    raise ValueError("Could not find a discharge disposition/location column in admissions.csv")

# ==========================================================
# 3. DEFINE RACE MAPPING FUNCTION
#    (race comes from ADMISSIONS -> 'race' column)
# ==========================================================

def map_race(raw_value):
    """Map detailed race/ethnicity strings into broad groups."""
    if pd.isna(raw_value):
        return "Other/Unknown"

    s = str(raw_value).lower()

    if "hispanic" in s or "latino" in s:
        return "Hispanic/Latino"
    if "black" in s or "african" in s:
        return "Black"
    if "white" in s:
        return "White"
    if (
        "asian" in s
        or "chinese" in s
        or "korean" in s
        or "japanese" in s
        or "filipino" in s
        or "vietnam" in s
    ):
        return "Asian"
    if "native" in s or "american indian" in s or "alaska" in s:
        return "Native American"
    return "Other/Unknown"

# ==========================================================
# 4. MERGE PATIENT DEMOGRAPHICS INTO ADMISSIONS
# ==========================================================

demo_cols = [SUBJECT_COL, AGE_COL, GENDER_COL]
df = admissions.merge(patients[demo_cols], on=SUBJECT_COL, how="left")

# Create race_group using the 'race' column from admissions (now in df)
if "race" in df.columns:
    df["race_group"] = df["race"].apply(map_race)
else:
    df["race_group"] = "Other/Unknown"  # fallback if race not present

# Convert times to datetime
df[ADMIT_COL] = pd.to_datetime(df[ADMIT_COL])
df[DISCH_COL] = pd.to_datetime(df[DISCH_COL])

# Drop rows with missing times
df = df.dropna(subset=[ADMIT_COL, DISCH_COL])

# Compute length of stay in days
df["length_of_stay"] = (df[DISCH_COL] - df[ADMIT_COL]).dt.total_seconds() / (24 * 3600)

# ==========================================================
# 5. CREATE 30-DAY READMISSION LABEL
# ==========================================================

df = df.sort_values([SUBJECT_COL, ADMIT_COL])

# Next admission time for the same patient
df["next_admit"] = df.groupby(SUBJECT_COL)[ADMIT_COL].shift(-1)

# Days until next admission
df["days_to_next_admit"] = (df["next_admit"] - df[DISCH_COL]).dt.total_seconds() / (24 * 3600)

# Readmitted within 30 days? (1 = yes, 0 = no)
df["readmit_30d"] = (
    (df["days_to_next_admit"] >= 0) & (df["days_to_next_admit"] <= 30)
).astype(int)

print("Readmission label distribution (0=no, 1=yes):")
print(df["readmit_30d"].value_counts())

# ==========================================================
# 6. BUILD FEATURE MATRIX (X) AND TARGET (y)
# ==========================================================

target_col = "readmit_30d"

numeric_features = [AGE_COL, "length_of_stay"]
categorical_features = [GENDER_COL, "race_group", DISP_COL]

# Drop rows with any missing values in these columns
model_df = df[numeric_features + categorical_features + [target_col]].dropna()

X = model_df[numeric_features + categorical_features]
y = model_df[target_col]

print("Final dataset shape (rows, columns):", X.shape)

# ==========================================================
# 7. TRAIN/TEST SPLIT (80/20, STRATIFIED)
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ==========================================================
# 8. PREPROCESSING PIPELINE (SCALING + ONE-HOT ENCODING)
# ==========================================================

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ==========================================================
# 9. LOGISTIC REGRESSION MODEL
# ==========================================================

log_reg_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

print("\nTraining Logistic Regression...")
log_reg_pipeline.fit(X_train, y_train)

y_pred_log = log_reg_pipeline.predict(X_test)
y_prob_log = log_reg_pipeline.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
try:
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))
except Exception as e:
    print("ROC-AUC could not be computed:", e)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

# ==========================================================
# 10. RANDOM FOREST MODEL (TUNED TO BE FASTER)
# ==========================================================

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=100,        # fewer trees (faster)
            max_depth=8,            # limit depth so trees are smaller
            min_samples_leaf=50,    # each leaf needs at least 50 samples
            n_jobs=-1,              # use all CPU cores
            class_weight="balanced",
            random_state=42,
        )),
    ]
)

print("\nTraining Random Forest...")
rf_pipeline.fit(X_train, y_train)

y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
try:
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
except Exception as e:
    print("ROC-AUC could not be computed:", e)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

print("\nDone!")
