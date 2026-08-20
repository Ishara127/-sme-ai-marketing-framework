import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset load karanna
df = pd.read_csv('marketing_campaign_dataset.csv')

# Basic info
print("=" * 50)
print("DATASET INFO")
print("=" * 50)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print()

# First 5 rows
print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())
print()

# Column names
print("=" * 50)
print("COLUMNS")
print("=" * 50)
print(df.columns.tolist())
print()

# Basic statistics
print("=" * 50)
print("BASIC STATISTICS")
print("=" * 50)
print(df.describe())
print()

# Missing values check
print("=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print()
print("Analysis complete! Dataset successfully loaded.")