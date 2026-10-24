from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset (changed to CSV format)
file_path = Path("diabetes_dataset.csv")  
if not file_path.exists():
  raise FileNotFoundError(
      f"CSV dataset not found! Place '{file_path.name}' in the project folder."
  )

df = pd.read_csv(file_path)
print(f"Original dataset shape: {df.shape}")

# 2. Define target and features
target = "Incident DM 0/1"
features = [
    "Sex, 1:Men, 2:Women",
    "Baseline Age, yrs",
    "Baseline BMI, kg/m2",
    "Baseline Waist circumference(WC), cm",
    "Baseline Habit of exercise 0/1",
    "Baseline Alcohol consumption, non:1, light:2, moderate:3, heavy:4",
    "Baseline Smoking status, 1:never, 2:past, 3:current",
]

missing = [c for c in features + [target] if c not in df.columns]
if missing:
  raise ValueError(f"Missing required columns: {missing}")

# 3. Clean dataset: select subset, drop missing values, filter valid binary targets
data = df[features + [target]].dropna(subset=features + [target]).copy()
data = data[data[target].isin([0, 1])].copy()
data[target] = data[target].astype(int)

print(f"Cleaned dataset shape: {data.shape}")
print("\nOutcome Distribution & Percentages:")
print(data[target].value_counts())
print(
    (data[target].value_counts(normalize=True) * 100).round(2).astype(str) + "%"
)

# 4. Separate features and target
X = data[features]
y = data[target]

# 5. Split into training and testing sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(
    f"\nSplit complete -> Training samples: {len(X_train)}, Testing samples:"
    f" {len(X_test)}"
)

# 6. Save prepared data to CSV files
data.to_csv("cleaned_diabetes_dataset.csv", index=False)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nAll prepared files saved successfully! Step 2 Completed.")