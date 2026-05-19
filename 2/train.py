# train.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("star_classification.csv")

# gunakan fitur fotometri
features = ['u', 'g', 'r', 'i', 'z']
X = df[features]

y = df['class']

# =========================================================
# LABEL ENCODING
# =========================================================

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================================================
# MODEL
# =========================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# EVALUATION
# =========================================================

preds = model.predict(X_test)

print(classification_report(y_test, preds))

# =========================================================
# SAVE MODEL
# =========================================================

import os

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/xgb_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("Model saved.")