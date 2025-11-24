import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

df = pd.read_csv("output/cleaned_dataset.csv")
print("Loaded cleaned dataset:", df.shape)

# Make sure the label exists
assert "readmitted_within_30d" in df.columns, "Missing readmitted_within_30d column"


# 1. T-TEST: Age vs Readmission (age difference between groups)

print("\n==============================")
print("T-TEST: Age vs Readmission (readmitted vs not)")
print("==============================")

age_readmitted = df[df["readmitted_within_30d"] == 1]["age"]
age_not_readmitted = df[df["readmitted_within_30d"] == 0]["age"]

t_stat, p_ttest = ttest_ind(age_readmitted, age_not_readmitted, equal_var=False)


print(f"T-statistic: {t_stat:.3f}")
print(f"p-value: {p_ttest:.4f}")
if p_ttest < 0.05:
    print("→ There IS a statistically significant difference in age between readmitted and non-readmitted patients.")
else:
    print("→ There is NO statistically significant difference in age between the two groups.")


# ============================================================
# 2. ANOVA: Age Group vs Readmission (0–40, 40–65, 65+)
# ============================================================

print("\n==============================")
print("ANOVA: Age Group vs Readmission")
print("==============================")

bins = [0, 40, 65, 200]
labels = ["0-40", "40-65", "65+"]

df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

g1 = df[df["age_group"] == "0-40"]["readmitted_within_30d"]
g2 = df[df["age_group"] == "40-65"]["readmitted_within_30d"]
g3 = df[df["age_group"] == "65+"]["readmitted_within_30d"]

f_stat, p_anova = f_oneway(g1, g2, g3)

print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_anova:.4f}")
if p_anova < 0.05:
    print("→ There IS a statistically significant difference in readmission rates across age groups.")
else:
    print("→ There is NO statistically significant difference in readmission rates across age groups.")


# ============================================================
# 3. CHI-SQUARE: Discharge Disposition vs Readmission
# (using each discharge_location_* dummy as 2x2)
# ============================================================

print("\n==============================")
print("CHI-SQUARE: Discharge Disposition vs Readmission")
print("==============================")

discharge_cols = [c for c in df.columns if c.startswith("discharge_location_")]
print("Discharge dummy columns:", discharge_cols)

for col in discharge_cols:
    print(f"\n--- {col} ---")
    # contingency table: membership (0/1 or False/True) vs readmission (0/1)
    table = pd.crosstab(df[col], df["readmitted_within_30d"])
    chi2, p_chi, dof, expected = chi2_contingency(table)

    print("Contingency table:")
    print(table)
    print(f"Chi-square: {chi2:.3f}")
    print(f"p-value: {p_chi:.4f}")

    if p_chi < 0.05:
        print("→ Significant association between this discharge category and readmission.")
    else:
        print("→ No significant association for this category.")


# ============================================================
# 4. CHI-SQUARE: Gender vs Readmission
# (using gender_* dummy columns)
# ============================================================

print("\n==============================")
print("CHI-SQUARE: Gender vs Readmission")
print("==============================")

gender_cols = [c for c in df.columns if c.startswith("gender_")]
print("Gender dummy columns:", gender_cols)

for col in gender_cols:
    print(f"\n--- {col} ---")
    table = pd.crosstab(df[col], df["readmitted_within_30d"])
    chi2, p_chi, dof, expected = chi2_contingency(table)

    print("Contingency table:")
    print(table)
    print(f"Chi-square: {chi2:.3f}")
    print(f"p-value: {p_chi:.4f}")

    if p_chi < 0.05:
        print("→ Significant association between this gender group and readmission.")
    else:
        print("→ No significant association for this gender group.")


# ============================================================
# 5. CHI-SQUARE: Race (Broad Groups) vs Readmission
# (using race_broad_* dummy columns)
# ============================================================

print("\n==============================")
print("CHI-SQUARE: Race (Broad Groups) vs Readmission")
print("==============================")

race_cols = [c for c in df.columns if c.startswith("race_broad_")]
print("Race dummy columns:", race_cols)

for col in race_cols:
    print(f"\n--- {col} ---")
    table = pd.crosstab(df[col], df["readmitted_within_30d"])
    chi2, p_chi, dof, expected = chi2_contingency(table)

    print("Contingency table:")
    print(table)
    print(f"Chi-square: {chi2:.3f}")
    print(f"p-value: {p_chi:.4f}")

    if p_chi < 0.05:
        print("→ Significant association between this race group and readmission.")
    else:
        print("→ No significant association for this race group.")
