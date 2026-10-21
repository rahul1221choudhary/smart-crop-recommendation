"""Train, compare and save the crop recommendation model.  Run: python train_model.py"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "Crop_recommendation.csv"
ART = ROOT / "artifacts"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

if not DATA.exists():
    sys.exit(f"\n[ERROR] Dataset not found: {DATA}\nSee data/README_DATASET.txt for download steps.\n"
             "No model was trained (this project never trains on fabricated data).")

# ---------- 1-9. Load + inspect ----------
df = pd.read_csv(DATA)
missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    sys.exit(f"[ERROR] Missing required columns: {missing_cols}. Found: {list(df.columns)}")
df = df[FEATURES + [TARGET]].copy()
df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

print("Shape:", df.shape)
print("\nFirst 5 records:\n", df.head())
print("\nData types:\n", df.dtypes)
n_missing = df.isna().sum()
print("\nMissing values:\n", n_missing)
n_dup = int(df.duplicated().sum())
print("Duplicate rows:", n_dup)
print("\nLabel distribution:\n", df[TARGET].value_counts())

invalid_mask = (
    (df[["N", "P", "K", "rainfall"]] < 0).any(axis=1)
    | ~df["ph"].between(0, 14)
    | ~df["humidity"].between(0, 100)
)
n_invalid = int(invalid_mask.sum())
print("Invalid numeric rows (negative N/P/K/rainfall, pH outside 0-14, humidity outside 0-100):", n_invalid)

rows_raw = len(df)
df = df[~invalid_mask].dropna().drop_duplicates().reset_index(drop=True)
print(f"Rows after removing missing/invalid/duplicate: {len(df)} (from {rows_raw})")

# ---------- 10-11. Features/target + split (before any preprocessing) ----------
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

# ---------- 12-13. Models (scaler lives INSIDE the pipeline -> no leakage) ----------
models = {
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
    "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1))]),
    "K-Nearest Neighbors": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for name, pipe in models.items():
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    rows.append({"Model": name, "CV F1 (train, 5-fold)": float(cv_f1.mean()),
                 "Accuracy": float(accuracy_score(y_test, pred)),
                 "Precision (macro)": float(p), "Recall (macro)": float(r), "F1-score (macro)": float(f)})
    fitted[name] = (pipe, pred)

comp = pd.DataFrame(rows)
print("\nModel comparison:\n", comp.round(4).to_string(index=False))

# Select by cross-validation on TRAINING data only (test set stays untouched for final reporting)
best_name = comp.sort_values("CV F1 (train, 5-fold)", ascending=False).iloc[0]["Model"]
best_pipe, best_pred = fitted[best_name]
print(f"\nSelected model: {best_name}")

# ---------- 14. Final evaluation ----------
labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)
report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
best_row = comp[comp["Model"] == best_name].iloc[0]

# ---------- 15-16. Save ----------
ART.mkdir(exist_ok=True)
joblib.dump(best_pipe, ART / "model.joblib")
meta = {
    "selected_model": best_name,
    "features": FEATURES,
    "labels": labels,
    "n_classes": len(labels),
    "rows_raw": rows_raw,
    "rows_clean": len(df),
    "duplicates_found": n_dup,
    "invalid_found": n_invalid,
    "missing_found": int(n_missing.sum()),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "seed": SEED,
    "test_accuracy": float(best_row["Accuracy"]),
    "comparison": comp.to_dict(orient="records"),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "feature_ranges": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                           "median": float(df[c].median())} for c in FEATURES},
}
(ART / "metadata.json").write_text(json.dumps(meta, indent=2))
print("\nSaved artifacts/model.joblib and artifacts/metadata.json")
print(f"Test accuracy of selected model: {meta['test_accuracy']:.4f}")
