import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("output/cleaned_dataset.csv")

print("Loaded cleaned dataset:", df.shape)
print(df.head())

# ============================================================
# Overall 30-day readmission rate
# ============================================================
print("\n=== Overall 30-Day Readmission Rate ===")
overall_rate = (df["readmitted_within_30d"].mean() * 100).round(1)
print(f"{overall_rate}%")


# ============================================================
# Readmission by Discharge Location (dummy columns)
# ============================================================
print("\n=== Readmission by Discharge Location ===")

discharge_cols = [c for c in df.columns if c.startswith("discharge_location_")]

for col in discharge_cols:
    results = (
        df.groupby(col)["readmitted_within_30d"]
        .mean()
        .mul(100)
        .round(1)
        .rename("Readmission (%)")
        .reset_index()
    )

    # Handle boolean or 0/1 dummy values
    results[col] = results[col].replace({
        0: "No", 1: "Yes",
        False: "No", True: "Yes"
    })

    # Rename for nicer display
    results = results.rename(columns={col: "In this discharge category?"})

    print(f"\n{col}:")
    print(results)


# ============================================================
# Readmission by Age Group
# ============================================================
print("\n=== Readmission by Age Group ===")

bins = [0, 40, 65, 200]
labels = ["0-40", "40-65", "65+"]

df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

age_results = (
    df.groupby("age_group", observed=False)["readmitted_within_30d"]
    .mean()
    .mul(100)
    .round(1)
    .rename("Readmission (%)")
    .reset_index()
)

print(age_results)


# ============================================================
# Readmission by Gender (dummy columns)
# ============================================================
print("\n=== Readmission by Gender ===")

gender_cols = [c for c in df.columns if c.startswith("gender_")]

for col in gender_cols:
    results = (
        df.groupby(col)["readmitted_within_30d"]
        .mean()
        .mul(100)
        .round(1)
        .rename("Readmission (%)")
        .reset_index()
    )

    results[col] = results[col].replace({
        0: "No", 1: "Yes",
        False: "No", True: "Yes"
    })

    results = results.rename(columns={col: "In this gender group?"})

    print(f"\n{col}:")
    print(results)


# ============================================================
# Readmission by Race (Broad Groups, dummy columns)
# ============================================================
print("\n=== Readmission by Race (Broad Groups) ===")

race_cols = [c for c in df.columns if c.startswith("race_broad_")]

for col in race_cols:
    results = (
        df.groupby(col)["readmitted_within_30d"]
        .mean()
        .mul(100)
        .round(1)
        .rename("Readmission (%)")
        .reset_index()
    )

    results[col] = results[col].replace({
        0: "Not in group", 1: "In group",
        False: "Not in group", True: "In group"
    })

    results = results.rename(columns={col: "Group membership"})

    print(f"\n{col}:")
    print(results)
