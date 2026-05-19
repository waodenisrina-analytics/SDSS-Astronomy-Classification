# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

from xgboost import XGBClassifier

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("star_classification.csv")

# =========================================================
# FEATURE ENGINEERING
# =========================================================

# =========================================================
# 1. FLAG NILAI MELEBIHI AMBANG BATAS
# =========================================================

df['u_limit_flag'] = (df['u'] > 24.5).astype(int)
df['r_limit_flag'] = (df['r'] > 24.5).astype(int)

# =========================================================
# 2. COLOR INDEX FEATURES
# =========================================================

df['u-g'] = df['u'] - df['g']
df['g-r'] = df['g'] - df['r']
df['r-i'] = df['r'] - df['i']
df['i-z'] = df['i'] - df['z']

# =========================================================
# 3. PCA FEATURES
# =========================================================

features_color = ['u-g', 'g-r', 'r-i', 'i-z']

X_color = df[features_color].values

pca_scaler = StandardScaler()
X_color_scaled = pca_scaler.fit_transform(X_color)

pca = PCA(n_components=3)

pca_features = pca.fit_transform(X_color_scaled)

df['PC1'] = pca_features[:, 0]
df['PC2'] = pca_features[:, 1]
df['PC3'] = pca_features[:, 2]

# =========================================================
# 4. DISTANCE FROM STELLAR LOCUS
# =========================================================

df_stars = df[df['class'] == 'STAR']

X_star_ug = df_stars['u-g']
y_star_gr = df_stars['g-r']

poly_coeffs = np.polyfit(X_star_ug, y_star_gr, deg=3)

locus_function = np.poly1d(poly_coeffs)

df['dist_locus'] = np.abs(
    df['g-r'] - locus_function(df['u-g'])
)

# =========================================================
# FINAL FEATURES
# =========================================================

fitur_final = [
    'r',
    'u-g',
    'g-r',
    'r-i',
    'i-z',
    'PC1',
    'PC2',
    'PC3',
    'dist_locus',
    'u_limit_flag',
    'r_limit_flag'
]

X = df[fitur_final]

# =========================================================
# TARGET
# =========================================================

y_raw = df['class']

encoder = LabelEncoder()

y = encoder.fit_transform(y_raw)

print(f"Mapping Class:")
print(dict(enumerate(encoder.classes_)))

# =========================================================
# SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# MODEL
# =========================================================

model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)

# =========================================================
# TRAINING
# =========================================================

print("\nTraining XGBoost Model...\n")

model.fit(X_train_scaled, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test_scaled)

# =========================================================
# EVALUATION
# =========================================================

print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

# =========================================================
# SAVE ARTIFACTS
# =========================================================

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/xgb_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")
joblib.dump(pca_scaler, "model/pca_scaler.pkl")
joblib.dump(pca, "model/pca.pkl")
joblib.dump(poly_coeffs, "model/locus_coeffs.pkl")

print("\nAll model artifacts saved.")