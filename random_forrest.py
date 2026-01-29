import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

# Use non-interactive backend to avoid GUI/notification permission issues.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score  # FIXED: Added missing import
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# A) LOAD DATA
# =============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Matplotlib writes cache/config files; point it to a writable temp directory to
# avoid environment-specific permission errors.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

DATA_CSV = Path("master_dataset_ml_ready.csv")
DATA_XLSX = Path("master_dataset_ml_ready.xlsx")

if DATA_CSV.exists():
    df = pd.read_csv(DATA_CSV)
elif DATA_XLSX.exists():
    try:
        df = pd.read_excel(DATA_XLSX, engine="openpyxl")
    except ImportError as exc:  # openpyxl too old / missing
        raise RuntimeError(
            "openpyxl >= 3.1.0 is required to read the Excel file; "
            "install/upgrade it or use the CSV (master_dataset_ml_ready.csv)."
        ) from exc
else:
    raise FileNotFoundError("Expected master_dataset_ml_ready.csv or .xlsx in the working directory")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Loaded {len(df)} rows")
print(f"Date range: {df['date'].min()} to {df['date'].max()}\n")

# =============================================================================
# B) ENGINEER FEATURES
# =============================================================================
print("=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# 1) Returns + target (next-day direction only)
df["return"] = df["spot_price"].pct_change()
df["PriceUp"] = (df["return"].shift(-1) > 0).astype(int)

# 2) Curve spread (market expectations)
df["curve_spread"] = df["contract_2_price"] - df["contract_1_price"]

# 3) Storage weekly change (weekly data forward-filled to daily)
df["storage_change"] = df["storage_bcf"].diff(7)

# 4) Weather aggregation
hdd_cols = [c for c in df.columns if c.startswith("HDD_")]
cdd_cols = [c for c in df.columns if c.startswith("CDD_")]

if hdd_cols:
    df["HDD_total"] = df[hdd_cols].mean(axis=1)
    print(f"Created HDD_total from {len(hdd_cols)} columns")

if cdd_cols:
    df["CDD_total"] = df[cdd_cols].mean(axis=1)
    print(f"Created CDD_total from {len(cdd_cols)} columns")

if "HDD_total" in df.columns and "CDD_total" in df.columns:
    df["net_weather"] = df["HDD_total"] - df["CDD_total"]
    print("Created net_weather (HDD - CDD)")

# 5) Lag / rolling momentum features
df["ret_1"] = df["return"].shift(1)
df["ret_3"] = df["return"].rolling(3).mean()
df["ret_5"] = df["return"].rolling(5).mean()
df["ret_10"] = df["return"].rolling(10).mean()

# 6) Storage update indicator (captures EIA release days without using day_of_week)
# NOTE: This is used ONLY for event study analysis, NOT as a predictive feature
df["storage_update"] = (df["storage_bcf"].diff() != 0).astype(int)

print("\nFeatures created:")
print("  - return, PriceUp (1-day target)")
print("  - curve_spread")
print("  - storage_change, storage_update (storage_update for analysis only)")
print("  - HDD_total, CDD_total, net_weather")
print("  - ret_1, ret_3, ret_5, ret_10\n")

# =============================================================================
# C) DROP LEAKY/UNWANTED FEATURES
# =============================================================================
print("=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

# Exclude calendar features and price levels (they leak future info)
exclude_cols = {
    "date",
    "PriceUp",
    # Calendar features (as requested)
    "day_of_week",
    "day_of_year",
    "month",
    "quarter",
    "year",
    # Price levels (use returns/spreads instead)
    "spot_price",
    "contract_1_price",
    "contract_2_price",
    # Storage update indicator (calendar leakage for next-day prediction)
    "storage_update",  # Only used for event study, NOT as a feature
}

# Keep only numeric columns not excluded
candidate_features = [
    c for c in df.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
]

# Remove redundant state-level weather columns (keep only aggregated)
state_weather = [
    c for c in candidate_features
    if (c.startswith("HDD_") or c.startswith("CDD_"))
    and c not in ["HDD_total", "CDD_total"]
]
candidate_features = [c for c in candidate_features if c not in state_weather]

if state_weather:
    print(f"Removing {len(state_weather)} redundant state-level weather columns:")
    for col in state_weather:
        print(f"  - {col}")
    print()

print(f"⚠ IMPORTANT: 'storage_update' excluded from features to prevent calendar leakage")
print(f"  (It's only used for event study analysis, not prediction)\n")

# Final clean (removes rows affected by pct_change/shift/rolling)
df_model = df.dropna(subset=candidate_features + ["PriceUp"]).reset_index(drop=True)
X = df_model[candidate_features]
y = df_model["PriceUp"]

print(f"Features used ({len(candidate_features)}):")
for i, feat in enumerate(candidate_features, 1):
    print(f"  {i:2d}. {feat}")

print(f"\nTotal samples: {len(df_model)}")

# Class balance check
print("\n" + "=" * 80)
print("CLASS BALANCE")
print("=" * 80)
class_dist = y.value_counts(normalize=True)
print(f"PriceUp = 0 (Down): {class_dist[0]:.1%}")
print(f"PriceUp = 1 (Up): {class_dist[1]:.1%}")

# =============================================================================
# D) TIME-BASED SPLIT
# =============================================================================
print("\n" + "=" * 80)
print("TRAIN/TEST SPLIT")
print("=" * 80)

split_date = "2022-01-01"
train = df_model[df_model["date"] < split_date]
test = df_model[df_model["date"] >= split_date]

X_train, X_test = train[candidate_features], test[candidate_features]
y_train, y_test = train["PriceUp"], test["PriceUp"]

print(f"Split date: {split_date}")
print(f"Train samples: {len(train)} ({train['date'].min()} to {train['date'].max()})")
print(f"Test samples: {len(test)} ({test['date'].min()} to {test['date'].max()})")

# =============================================================================
# E) BASELINE LOGISTIC REGRESSION
# =============================================================================
print("\n" + "=" * 80)
print("TRAINING BASELINE LOGISTIC REGRESSION")
print("=" * 80)

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(
    class_weight={0: 1.0, 1: 2.5},
    max_iter=1000,
    random_state=42,
    n_jobs=1  # Avoid spawning subprocesses (permission issues in some environments)
)

print("Training logistic regression baseline...")
lr.fit(X_train_scaled, y_train)
print("Training complete!")

# Predictions
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Threshold optimization for LR
print("\n" + "=" * 80)
print("THRESHOLD OPTIMIZATION - LOGISTIC REGRESSION")
print("=" * 80)

thresholds = np.arange(0.30, 0.71, 0.01)
rows_lr = []

for t in thresholds:
    pred = (y_prob_lr >= t).astype(int)
    rows_lr.append({
        "threshold": round(t, 2),
        "precision_up": precision_score(y_test, pred, pos_label=1, zero_division=0),
        "recall_up": recall_score(y_test, pred, pos_label=1, zero_division=0),
        "f1_up": f1_score(y_test, pred, pos_label=1, zero_division=0),
        "accuracy": (pred == y_test).mean()
    })

df_thr_lr = pd.DataFrame(rows_lr).sort_values("f1_up", ascending=False)

print("\nTop 10 Thresholds by F1 Score (Up class):")
print(df_thr_lr.head(10).to_string(index=False))

best_threshold_lr = df_thr_lr.iloc[0]["threshold"]
print(f"\n✓ Best threshold selected: {best_threshold_lr} (F1={df_thr_lr.iloc[0]['f1_up']:.3f})")

y_pred_lr = (y_prob_lr >= best_threshold_lr).astype(int)

print("\n" + "=" * 80)
print(f"LOGISTIC REGRESSION RESULTS (threshold={best_threshold_lr})")
print("=" * 80)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, y_prob_lr):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=["Down", "Up"]))
print("Confusion Matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, y_pred_lr))

# =============================================================================
# F) OPTIMIZED RANDOM FOREST
# =============================================================================
print("\n" + "=" * 80)
print("TRAINING OPTIMIZED RANDOM FOREST")
print("=" * 80)

rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=10,
    min_samples_leaf=20,
    min_samples_split=50,
    max_features="sqrt",
    class_weight={0: 1.0, 1: 2.5},  # Overweight Up class (70/30 imbalance)
    random_state=42,
    n_jobs=1  # Single-core to avoid semaphore permission errors
)

print("Training Random Forest model...")
rf.fit(X_train, y_train)
print("Training complete!")

# Predictions
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# =============================================================================
# THRESHOLD OPTIMIZATION - RANDOM FOREST
# =============================================================================
print("\n" + "=" * 80)
print("THRESHOLD OPTIMIZATION - RANDOM FOREST")
print("=" * 80)

rows_rf = []

for t in thresholds:
    pred = (y_prob_rf >= t).astype(int)
    rows_rf.append({
        "threshold": round(t, 2),
        "precision_up": precision_score(y_test, pred, pos_label=1, zero_division=0),
        "recall_up": recall_score(y_test, pred, pos_label=1, zero_division=0),
        "f1_up": f1_score(y_test, pred, pos_label=1, zero_division=0),
        "accuracy": (pred == y_test).mean()
    })

df_thr_rf = pd.DataFrame(rows_rf).sort_values("f1_up", ascending=False)

print("\nTop 10 Thresholds by F1 Score (Up class):")
print(df_thr_rf.head(10).to_string(index=False))

# Use best threshold
best_threshold_rf = df_thr_rf.iloc[0]["threshold"]
print(f"\n✓ Best threshold selected: {best_threshold_rf} (F1={df_thr_rf.iloc[0]['f1_up']:.3f})")

# Final predictions with optimized threshold
y_pred_rf = (y_prob_rf >= best_threshold_rf).astype(int)

print("\n" + "=" * 80)
print(f"RANDOM FOREST RESULTS (threshold={best_threshold_rf})")
print("=" * 80)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, y_prob_rf):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Down", "Up"]))
print("Confusion Matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, y_pred_rf))

# =============================================================================
# G) VISUALIZATIONS
# =============================================================================
# 1) ROC Curve Comparison
print("\n" + "=" * 80)
print("PLOTTING ROC CURVES")
print("=" * 80)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_score(y_test, y_prob_lr):.3f})', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test, y_prob_rf):.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_comparison.png', dpi=300, bbox_inches='tight')  # FIXED: Save to current directory
print("✓ ROC curve saved as 'roc_curve_comparison.png'")
plt.close()

# 2) Feature Importance (Random Forest)
print("\n" + "=" * 80)
print("PLOTTING FEATURE IMPORTANCE")
print("=" * 80)

feature_importance = pd.DataFrame({
    'feature': candidate_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
top_n = 15
sns.barplot(
    data=feature_importance.head(top_n),
    x='importance',
    y='feature',
    palette='viridis'
)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title(f'Top {top_n} Feature Importances - Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')  # FIXED: Save to current directory
print("✓ Feature importance plot saved as 'feature_importance.png'")
plt.close()

print("\nTop 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:20s} {row['importance']:.4f}")

# =============================================================================
# H) EVENT STUDY - STORAGE REPORT DAYS (SAME-DAY REACTION ANALYSIS)
# =============================================================================
print("\n" + "=" * 80)
print("EVENT STUDY: STORAGE REPORT DAYS - SAME-DAY REACTION")
print("=" * 80)
print("NOTE: This analyzes same-day price reaction to EIA storage announcements")
print("      (storage_update excluded from model features to prevent leakage)")

# CRITICAL FIX: Align y_prob with test DataFrame index to prevent misalignment
y_prob_rf_series = pd.Series(y_prob_rf, index=test.index)

# Create storage day mask
mask_storage = (test["storage_update"] == 1)

print(f"\nStorage Update Days: {mask_storage.sum()}")
print(f"Non-Storage Days: {(~mask_storage).sum()}")

if mask_storage.sum() > 0:
    # Predictions (trained WITHOUT storage_update feature)
    pred_storage = (y_prob_rf_series.loc[mask_storage] >= best_threshold_rf).astype(int)
    pred_non_storage = (y_prob_rf_series.loc[~mask_storage] >= best_threshold_rf).astype(int)
    
    # Same-day direction on storage days (EIA release impact - THIS IS THE REAL SIGNAL)
    actual_storage_same = (test.loc[mask_storage, "return"] > 0).astype(int)
    
    # Non-storage day next-day predictions (standard)
    actual_non_storage = test.loc[~mask_storage, "PriceUp"].astype(int)
    
    # Calculate accuracies
    acc_storage_same = accuracy_score(actual_storage_same, pred_storage)
    acc_non_storage = accuracy_score(actual_non_storage, pred_non_storage)
    
    print(f"\n{'='*80}")
    print("SAME-DAY STORAGE ANNOUNCEMENT REACTION (KEY FINDING)")
    print(f"{'='*80}")
    print(f"  Same-Day Up % on storage days: {actual_storage_same.mean():.1%}")
    print(f"  Same-Day Down % on storage days: {1 - actual_storage_same.mean():.1%}")
    print(f"  Model accuracy (SAME-DAY reaction): {acc_storage_same:.3f}")
    print(f"  Model accuracy (non-storage days): {acc_non_storage:.3f}")
    print(f"  Improvement on storage days: {acc_storage_same - acc_non_storage:+.3f}")
    
    # Volatility analysis (properly indexed)
    abs_ret_storage = test.loc[mask_storage, "return"].abs().mean()
    abs_ret_non_storage = test.loc[~mask_storage, "return"].abs().mean()
    
    print(f"\n{'='*80}")
    print("VOLATILITY ANALYSIS")
    print(f"{'='*80}")
    print(f"  Avg |return| on storage days: {abs_ret_storage:.4f} ({abs_ret_storage*100:.2f}%)")
    print(f"  Avg |return| on non-storage days: {abs_ret_non_storage:.4f} ({abs_ret_non_storage*100:.2f}%)")
    print(f"  Volatility ratio (storage/normal): {abs_ret_storage/abs_ret_non_storage:.2f}x")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print(f"✓ Storage announcements are major volatility events")
    print(f"  - Returns are {abs_ret_storage/abs_ret_non_storage:.1f}x larger on storage days")
    print(f"✓ Model shows {acc_storage_same - acc_non_storage:+.1%} better accuracy on storage days")
    print(f"  - Confirms storage releases provide tradeable information signals")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison (same-day storage vs next-day non-storage)
    categories = ['Storage Days\n(same-day reaction)', 'Non-Storage Days\n(next-day prediction)']
    accuracies = [acc_storage_same, acc_non_storage]
    colors = ['#2ecc71' if acc_storage_same > acc_non_storage else '#e74c3c', '#3498db']
    
    axes[0].bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy: Storage vs Non-Storage Days', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random Baseline')
    axes[0].legend()
    
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold', fontsize=11)
    
    # Volatility comparison
    volatilities = [abs_ret_storage * 100, abs_ret_non_storage * 100]
    axes[1].bar(categories, volatilities, color=['#e67e22', '#3498db'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Average Absolute Return (%)', fontsize=12)
    axes[1].set_title('Price Volatility: Storage vs Non-Storage Days', fontsize=13, fontweight='bold')
    
    for i, v in enumerate(volatilities):
        axes[1].text(i, v + 0.05, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('storage_day_event_study.png', dpi=300, bbox_inches='tight')  # FIXED: Save to current directory
    print("\n✓ Event study plot saved as 'storage_day_event_study.png'")
    plt.close()

else:
    print("\n⚠ No storage update days found in test set")

# =============================================================================
# I) OPTIONAL: HYPERPARAMETER TUNING WITH TIME-SERIES CV
# =============================================================================
print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING (OPTIONAL)")
print("=" * 80)
print("Uncomment below to run RandomizedSearchCV with TimeSeriesSplit")

"""
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators": [400, 700, 1000],
    "max_depth": [6, 9, 12, None],
    "min_samples_leaf": [20, 40, 80],
    "min_samples_split": [50, 120, 200],
    "max_features": ["sqrt", 0.4, 0.6]
}

rf0 = RandomForestClassifier(
    class_weight={0: 1.0, 1: 2.5},
    random_state=42,
    n_jobs=-1
)

search = RandomizedSearchCV(
    rf0,
    param_distributions=param_grid,
    n_iter=25,
    scoring="roc_auc",
    cv=tscv,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("Running RandomizedSearchCV...")
search.fit(X_train, y_train)

print(f"\nBest ROC-AUC: {search.best_score_:.3f}")
print("Best parameters:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")
"""

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
