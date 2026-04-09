# ==============================
# Statistics Application in Genetics & Genomics
# Dataset: Cell_Lines_Details.xlsx
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------
# 1. Load the dataset
# ------------------------------
df = pd.read_excel("Cell_Lines_Details (1).xlsx")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ------------------------------
# 2. Select genomic columns
# (column names may slightly differ; adjust if needed)
# ------------------------------
genomic_cols = [
    'WES',                 # Whole Exome Sequencing
    'CNA',                 # Copy Number Alterations
    'Gene Expression',
    'Methylation'
]

# ------------------------------
# 3. Convert Y/N to 1/0
# ------------------------------
for col in genomic_cols:
    df[col] = df[col].map({'Y': 1, 'N': 0})

# ------------------------------
# 4. Create numerical genomic score
# ------------------------------
df['Genomic_Score'] = df[genomic_cols].sum(axis=1)

print("\nSample Genomic Scores:")
print(df[['Sample Name', 'Genomic_Score']].head(10))

# ------------------------------
# 5. Mean, Median, Mode
# ------------------------------
mean_val = df['Genomic_Score'].mean()
median_val = df['Genomic_Score'].median()
mode_val = df['Genomic_Score'].mode()[0]

print("\n--- Central Tendency ---")
print("Mean Genomic Score:", round(mean_val, 2))
print("Median Genomic Score:", median_val)
print("Mode Genomic Score:", mode_val)

# ------------------------------
# 6. Variance & Standard Deviation
# ------------------------------
variance_val = df['Genomic_Score'].var()
std_dev_val = df['Genomic_Score'].std()

print("\n--- Dispersion ---")
print("Variance:", round(variance_val, 2))
print("Standard Deviation:", round(std_dev_val, 2))

# ------------------------------
# 7. Skewness
# ------------------------------
skewness_val = stats.skew(df['Genomic_Score'])

print("\n--- Distribution Shape ---")
print("Skewness:", round(skewness_val, 2))

# ------------------------------
# 8. Histogram (Skewness + Modality)
# ------------------------------
plt.figure()
plt.hist(df['Genomic_Score'], bins=5)
plt.xlabel("Genomic Score (0–4)")
plt.ylabel("Number of Cell Lines")
plt.title("Histogram of Genomic Score")
plt.show()

# ------------------------------
# 9. Bar Graph
# ------------------------------
df['Genomic_Score'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Genomic Score")
plt.ylabel("Number of Cell Lines")
plt.title("Bar Graph of Genomic Data Availability")
plt.show()

# ------------------------------
# 10. Scatter Plot
# ------------------------------
# Convert Drug Response to numeric
df['Drug Response'] = df['Drug Response'].map({'Y': 1, 'N': 0})

plt.figure()
plt.scatter(df['Genomic_Score'], df['Drug Response'])
plt.xlabel("Genomic Score")
plt.ylabel("Drug Response Availability (0/1)")
plt.title("Scatter Plot: Genomic Score vs Drug Response")
plt.show()

# ------------------------------
# 11. Print a few real data points for presentation
# ------------------------------
print("\n--- Example Data Points Used ---")
print(df[['Sample Name', 'Genomic_Score', 'Drug Response']].head(8))