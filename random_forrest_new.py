import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('master_dataset_ml_ready.csv')

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Create binary target: next-day spot price direction (1 = up, 0 = down)
df['PriceUp'] = (df['spot_price'].shift(-1) > df['spot_price']).astype(int)

# Define features and target
target = 'PriceUp'
exclude_cols = ['date', 'spot_price', 'day_of_week', target]

# Storage-day features (focus on weekly storage report days: 3rd & 4th day of week)
df['storage_day3'] = df['storage_bcf'] * (df['day_of_week'] == 3)
df['storage_day4'] = df['storage_bcf'] * (df['day_of_week'] == 4)

features = [col for col in df.columns if col not in exclude_cols]

print(f"\nFeatures used ({len(features)}):")
print(features)

# Remove rows with missing target or features
df_clean = df.dropna(subset=[target] + features)
print(f"\nDataset after removing NaN: {df_clean.shape}")

X = df_clean[features]
y = df_clean[target]

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
print(f"Class distribution - Test: {y_test.value_counts().to_dict()}")

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n" + "="*60)
print("RANDOM FOREST CLASSIFICATION RESULTS")
print("="*60)

print("\nTRAIN SET PERFORMANCE:")
print(classification_report(y_train, y_train_pred, target_names=['Down', 'Up']))

print("\nTEST SET PERFORMANCE:")
print(classification_report(y_test, y_test_pred, target_names=['Down', 'Up']))

print("\nCONFUSION MATRIX (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

test_auc = roc_auc_score(y_test, y_test_proba)
print(f"\nROC-AUC Score (Test): {test_auc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 15 FEATURE IMPORTANCE:")
print(feature_importance.head(15))

# Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Feature Importance
axes[0, 0].barh(feature_importance.head(15)['feature'], 
                feature_importance.head(15)['importance'])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Top 15 Feature Importance')
axes[0, 0].invert_yaxis()

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d',
            ax=axes[0, 1], cmap='Blues')
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[1, 0].plot(fpr, tpr, label=f'ROC-AUC = {test_auc:.4f}')
axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()

# Class Distribution
y_test.value_counts().plot(kind='bar', ax=axes[1, 1], color=['coral', 'skyblue'])
axes[1, 1].set_title('Test Set - PriceUp Distribution')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_xlabel('PriceUp')

plt.tight_layout()
plt.savefig('random_forest_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved as 'random_forest_analysis.png'")
