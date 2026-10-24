# inspect and eda
from pathlib import Path
import pandas as pd

# 1. Load the CSV dataset (change filename if needed)
FILE_PATH = Path("diabetes_dataset.csv")

if not FILE_PATH.exists():
  raise FileNotFoundError(
      f"Dataset not found! Place '{FILE_PATH.name}' in this directory."
  )

print("DIABETES RISK PREDICTION - DATASET INSPECTION")

# Read the CSV file directly
df = pd.read_csv(FILE_PATH)

# 2. Basic dataset properties
print(f"\nDimensions: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nColumns:")
for i, col in enumerate(df.columns, start=1):
  print(f"{i}. {col}")

# 3. Preview data & data types
print("\nFirst 5 Records:")
print(df.head().to_string())

print("\nData Types:")
print(df.dtypes.to_string())

# 4. Check quality issues (missing values & duplicates)
missing = df.isnull().sum()[df.isnull().sum() > 0]
print("\nMissing Values:")
print(missing.to_string() if not missing.empty else "None detected.")

print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# 5. Summary statistics
print("\nStatistical Summary:")
print(df.describe().to_string())

# 6. Target column inspection
print("\nDiabetes Outcome Target Search:")
targets = [
    col
    for col in df.columns
    if any(k in str(col).lower() for k in ["incident dm", "diabetes", "diabetic"])
]

if targets:
  for target in targets:
    print(f"\nTarget Column: {target}")
    print(df[target].value_counts(dropna=False).to_string())
else:
  print(
      "No outcome column matched keywords ('incident dm', 'diabetes',"
      " 'diabetic'). Check column list above."
  )

print("\nInspection Completed.")