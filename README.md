# 🌌 SDSS Astronomy Classification Dashboard

Explainable Machine Learning dashboard for astronomical object classification using SDSS photometric data.

## 🚀 Overview

This project classifies celestial objects into:

- GALAXY
- STAR
- QSO (Quasar)

using photometric measurements from the Sloan Digital Sky Survey (SDSS).

The system combines:

- Astronomy-aware feature engineering
- Stellar locus modelling
- PCA latent representation
- XGBoost classification
- Interactive Streamlit dashboard

---

## 🛰 Features

### Machine Learning
- XGBoost multi-class classification
- PCA feature compression
- Stellar locus distance modelling
- Scientific feature engineering

### Dashboard
- Interactive prediction simulator
- Feature importance visualization
- Stellar locus visualization
- Astronomy analytics interface

---

## 📊 Feature Engineering

The project includes:

- Color indices:
  - u-g
  - g-r
  - r-i
  - i-z

- PCA components:
  - PC1
  - PC2
  - PC3

- Stellar locus distance

- Threshold flag features

---

## 🤖 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 88.7% |
| Macro F1 | 85.5% |

---

## 🛠 Tech Stack

- Python
- Streamlit
- XGBoost
- Scikit-Learn
- Plotly
- Pandas
- NumPy

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py