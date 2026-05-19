# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SDSS Astronomy Classification",
    page_icon="🌌",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #050816;
    color: white;
}

.stApp {
    background: linear-gradient(
        180deg,
        #050816 0%,
        #0b1026 50%,
        #111936 100%
    );
}

h1, h2, h3 {
    color: #e2e8f0;
}

.metric-card {
    background-color: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.1);
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA & MODEL
# =========================================================

df = pd.read_csv("star_classification.csv")

model = joblib.load("model/xgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le = joblib.load("model/label_encoder.pkl")

features = ['u', 'g', 'r', 'i', 'z']

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌌 SDSS Astronomy ML")
page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Dataset Exploration",
        "Feature Analysis",
        "Model Performance",
        "Prediction Simulator"
    ]
)

# =========================================================
# PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.title("🌌 SDSS Stellar Object Classification")

    st.markdown("""
    ### Explainable Machine Learning for Astronomical Object Classification

    This project classifies astronomical objects into:

    - GALAXY
    - STAR
    - QSO (Quasar)

    using photometric measurements from the Sloan Digital Sky Survey (SDSS).
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Data", f"{len(df):,}")
    col2.metric("Features", "5")
    col3.metric("Classes", df['class'].nunique())
    col4.metric("Model", "XGBoost")

    st.divider()

    st.subheader("📡 Feature Description")

    feature_df = pd.DataFrame({
        "Feature": ['u', 'g', 'r', 'i', 'z'],
        "Description": [
            'Ultraviolet Magnitude',
            'Green Magnitude',
            'Red Magnitude',
            'Near Infrared Magnitude',
            'Infrared Magnitude'
        ]
    })

    st.dataframe(feature_df, use_container_width=True)

    st.divider()

    st.subheader("🛰 Machine Learning Pipeline")

    st.markdown("""
    ```text
    SDSS Data
        ↓
    Data Cleaning
        ↓
    Feature Engineering
        ↓
    Scaling
        ↓
    XGBoost Training
        ↓
    Model Evaluation
        ↓
    Interactive Prediction Dashboard
    ```
    """)

# =========================================================
# DATASET EXPLORATION
# =========================================================

elif page == "Dataset Exploration":

    st.title("📊 Dataset Exploration")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            df,
            names='class',
            title='Class Distribution',
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig2 = px.histogram(
            df,
            x='r',
            color='class',
            nbins=50,
            title='Red Band Distribution'
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Raw Dataset")

    st.dataframe(df.head(100), use_container_width=True)

# =========================================================
# FEATURE ANALYSIS
# =========================================================

elif page == "Feature Analysis":

    st.title("🔭 Feature Analysis")

    corr = df[features].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    selected_feature = st.selectbox(
        "Select Feature",
        features
    )

    fig2 = px.box(
        df,
        x='class',
        y=selected_feature,
        color='class',
        title=f'{selected_feature} Distribution by Class'
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='XGBoost Feature Importance'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Scientific Interpretation")

    st.markdown("""
    - Features with high importance indicate strong discriminative power.
    - Color-related photometric bands help separate galaxies, stars, and quasars.
    - Quasars typically exhibit unusual spectral energy distributions.
    - Galaxy observations often differ in infrared and red bands.
    """)

# =========================================================
# PREDICTION SIMULATOR
# =========================================================

elif page == "Prediction Simulator":

    st.title("🪐 Prediction Simulator")

    st.markdown("""
    Enter photometric magnitudes to predict astronomical object type.
    """)

    col1, col2 = st.columns(2)

    with col1:

        u = st.slider("u (Ultraviolet)", 10.0, 30.0, 18.0)
        g = st.slider("g (Green)", 10.0, 30.0, 17.0)
        r = st.slider("r (Red)", 10.0, 30.0, 16.0)

    with col2:

        i = st.slider("i (Near Infrared)", 10.0, 30.0, 15.0)
        z = st.slider("z (Infrared)", 10.0, 30.0, 14.0)

    input_data = np.array([[u, g, r, i, z]])

    scaled = scaler.transform(input_data)

    pred = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]

    pred_label = le.inverse_transform([pred])[0]

    st.divider()

    st.subheader("🔮 Prediction Result")

    st.success(f"Predicted Class: {pred_label}")

    confidence = np.max(probs) * 100

    st.metric("Confidence Score", f"{confidence:.2f}%")

    prob_df = pd.DataFrame({
        'Class': le.classes_,
        'Probability': probs
    })

    fig = px.bar(
        prob_df,
        x='Class',
        y='Probability',
        title='Prediction Probability Distribution'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Astrophysical Interpretation")

    if pred_label == "GALAXY":
        st.info("""
        The object shows photometric characteristics commonly associated with galaxies,
        particularly broader spectral distributions in red and infrared bands.
        """)

    elif pred_label == "STAR":
        st.info("""
        The object resembles stellar photometric signatures with smoother magnitude transitions.
        """)

    else:
        st.info("""
        The object resembles a quasar candidate with unusual photometric energy distribution.
        """)