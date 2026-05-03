import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
import joblib

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/fires.csv")

# Convert date
df['acq_date'] = pd.to_datetime(df['acq_date'])

# =========================
# 2. CREATE GRID
# =========================
df['lat_bin'] = df['latitude'].round(1)
df['lon_bin'] = df['longitude'].round(1)

# Aggregate fire detections
fires = df.groupby(['lat_bin', 'lon_bin', 'acq_date'], as_index=False).agg({
    'frp': 'mean'
})

# =========================
# 3. CREATE FULL GRID (CRITICAL FIX)
# =========================

# All dates
all_dates = pd.date_range(fires['acq_date'].min(), fires['acq_date'].max())

# All grid cells
grid_cells = fires[['lat_bin','lon_bin']].drop_duplicates()

# Create full grid
full_grid = grid_cells.merge(
    pd.DataFrame({'acq_date': all_dates}),
    how='cross'
)

# Merge fires
data = full_grid.merge(
    fires,
    on=['lat_bin','lon_bin','acq_date'],
    how='left'
)

# Create label
data['fire_today'] = data['frp'].notna().astype(int)

# Fill missing FRP
data['frp'] = data['frp'].fillna(0)

# =========================
# 4. SORT
# =========================
data = data.sort_values(['lat_bin','lon_bin','acq_date']).reset_index(drop=True)

# =========================
# 5. FEATURES
# =========================

# Lag
data['fire_last_1d'] = data.groupby(['lat_bin','lon_bin'])['fire_today'].shift(1)

# Rolling
data['fire_last_3d'] = data.groupby(['lat_bin','lon_bin'])['fire_today']\
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())

data['fire_last_7d'] = data.groupby(['lat_bin','lon_bin'])['fire_today']\
    .transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())

# =========================
# 6. TARGET
# =========================
data['target'] = data.groupby(['lat_bin','lon_bin'])['fire_today'].shift(-1)

# Drop NA
data = data.dropna().reset_index(drop=True)

# =========================
# 7. FEATURES & LABEL
# =========================
X = data[['lat_bin','lon_bin','frp','fire_last_1d','fire_last_3d','fire_last_7d']]
y = data['target']

# =========================
# 8. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================
# 9. TRAIN MODEL
# =========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=10,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# =========================
# 10. EVALUATION
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
print("PR-AUC   :", average_precision_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 11. SAVE MODEL
# =========================
joblib.dump(model, "model.pkl")

print("\nModel trained successfully!")