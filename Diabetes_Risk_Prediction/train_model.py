import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# 1. Load prepared data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]

print(
    f"Training shape: {X_train.shape} | Positive cases:"
    f" {int(y_train.sum())}"
)
print(
    f"Testing shape: {X_test.shape} | Positive cases: {int(y_test.sum())}"
)

# 2. Compute class imbalance weight for XGBoost
neg_cases = (y_train == 0).sum()
pos_cases = (y_train == 1).sum()
scale_weight = neg_cases / pos_cases
print(f"XGBoost positive-class weight: {scale_weight:.2f}")

# 3. Define models using pipelines
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=42
            ),
        ),
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=5,
                min_samples_leaf=20,
                random_state=42,
            ),
        ),
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]),
    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_weight,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]),
}

# 4. Train, evaluate, and save models
results = []
trained_models = {}

for name, model in models.items():
  print(f"\nTraining: {name}")
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:, 1]

  metrics = {
      "Model": name,
      "Accuracy": accuracy_score(y_test, y_pred),
      "Precision": precision_score(y_test, y_pred, zero_division=0),
      "Recall": recall_score(y_test, y_pred, zero_division=0),
      "F1 Score": f1_score(y_test, y_pred, zero_division=0),
      "ROC-AUC": roc_auc_score(y_test, y_prob),
      "PR-AUC": average_precision_score(y_test, y_prob),
  }
  results.append(metrics)
  trained_models[name] = model

  print(
      f"Acc: {metrics['Accuracy']:.4f} | Prec: {metrics['Precision']:.4f} | Rec:"
      f" {metrics['Recall']:.4f} | F1: {metrics['F1 Score']:.4f} | ROC-AUC:"
      f" {metrics['ROC-AUC']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f}"
  )
  print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5. Compare and export results
results_df = pd.DataFrame(results).sort_values(by="PR-AUC", ascending=False)
print("\nFINAL MODEL COMPARISON (Sorted by PR-AUC):")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

# 6. Save trained model binaries
for name, model in trained_models.items():
  filename = name.lower().replace(" ", "_") + ".joblib"
  joblib.dump(model, filename)
  print(f"Saved: {filename}")

# 7. Plot PR-AUC comparison chart
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["PR-AUC"], color="skyblue")
plt.title("Model Comparison Using PR-AUC")
plt.xlabel("Machine Learning Model")
plt.ylabel("Average Precision (PR-AUC)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

print("\nSTEP 3 COMPLETED SUCCESSFULLY!")