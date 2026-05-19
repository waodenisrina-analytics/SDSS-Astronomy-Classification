# =========================================================
# SDSS ASTRONOMY CLASSIFICATION DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

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

.stApp {
    background: linear-gradient(
        180deg,
        #050816 0%,
        #0b1026 50%,
        #111936 100%
    );
    color: white;
}

h1, h2, h3 {
    color: #e2e8f0;
}

[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 15px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("star_classification.csv")

# Variabel untuk di buang
dropped = ['obj_ID', 'alpha', 'delta', 'run_ID', 'rerun_ID', 
       'cam_col', 'field_ID', 'spec_obj_ID', 'redshift',
       'plate', 'MJD', 'fiber_ID']

# Pengambilan subset data
df = df.drop(columns=dropped)

mag_columns = ['u', 'g', 'r', 'i', 'z']
df[mag_columns] = df[mag_columns].mask(df[mag_columns] < 0, np.nan)

# Hapus nilai nan/anomali
df = df.dropna()

# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

model = joblib.load("model/xgb_model.pkl")

scaler = joblib.load("model/scaler.pkl")

encoder = joblib.load("model/label_encoder.pkl")

pca_scaler = joblib.load("model/pca_scaler.pkl")

pca = joblib.load("model/pca.pkl")

poly_coeffs = joblib.load("model/locus_coeffs.pkl")

locus_function = np.poly1d(poly_coeffs)

# =========================================================
# FEATURE ENGINEERING FOR VISUALIZATION
# =========================================================

df['u-g'] = df['u'] - df['g']
df['g-r'] = df['g'] - df['r']
df['r-i'] = df['r'] - df['i']
df['i-z'] = df['i'] - df['z']

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
    ## Explainable Machine Learning for Astronomy

    This project classifies astronomical objects into:

    - GALAXY
    - STAR
    - QSO (Quasar)

    using photometric measurements from the Sloan Digital Sky Survey (SDSS).

    The project focuses on:

    - Astronomy-aware feature engineering
    - Explainable machine learning
    - Photometric classification
    - Stellar locus analysis
    - Scientific visualization
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Data", f"{len(df):,}")
    col2.metric("Classes", df['class'].nunique())
    col3.metric("Features", "11")
    col4.metric("Model", "XGBoost")

    st.divider()

    st.subheader("📡 Features Used")

    fitur_df = pd.DataFrame({
        "Feature": [
            "r",
            "u-g",
            "g-r",
            "r-i",
            "i-z",
            "PC1",
            "PC2",
            "PC3",
            "dist_locus",
            "u_limit_flag",
            "r_limit_flag"
        ],
        "Description": [
            "Red magnitude",
            "Color index",
            "Color index",
            "Color index",
            "Color index",
            "Principal component 1",
            "Principal component 2",
            "Principal component 3",
            "Distance from stellar locus",
            "Ultraviolet threshold flag",
            "Red threshold flag"
        ]
    })

    st.dataframe(fitur_df, use_container_width=True)

    st.divider()

    st.subheader("🛰 Machine Learning Pipeline")

    st.markdown("""
    ```text
    SDSS Photometric Data
            ↓
    Color Feature Engineering
            ↓
    Stellar Locus Modelling
            ↓
    PCA Dimensional Compression
            ↓
    Feature Scaling
            ↓
    XGBoost Classification
            ↓
    Interactive Astronomy Dashboard
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
            nbins=60,
            title='Red Magnitude Distribution'
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(100), use_container_width=True)

# =========================================================
# FEATURE ANALYSIS
# =========================================================

elif page == "Feature Analysis":

    st.title("🔭 Feature Analysis")

    color_features = ['u-g', 'g-r', 'r-i', 'i-z']

    corr = df[color_features].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        title='Color Feature Correlation Matrix',
        aspect='auto'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("🌠 Stellar Locus Visualization")

    fig2 = px.scatter(
        df.sample(5000),
        x='u-g',
        y='g-r',
        color='class',
        opacity=0.7,
        title='Stellar Locus in Color Space'
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("📈 Feature Distribution")

    selected_feature = st.selectbox(
        "Select Feature",
        ['u-g', 'g-r', 'r-i', 'i-z']
    )

    fig3 = px.box(
        df,
        x='class',
        y=selected_feature,
        color='class',
        title=f'{selected_feature} Distribution'
    )

    st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

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

    importance_df = pd.DataFrame({
        'Feature': fitur_final,
        'Importance': model.feature_importances_
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

    st.subheader("📚 Scientific Interpretation")

    st.markdown("""
    ### Key Findings

    - Color indices are highly discriminative for astrophysical classification.
    - Stellar locus distance helps separate stars from quasars and galaxies.
    - PCA components capture latent photometric manifolds.
    - Quasars exhibit unique spectral energy distributions.
    - Galaxy observations dominate infrared/red photometric regions.
    """)

# =========================================================
# PREDICTION SIMULATOR
# =========================================================

elif page == "Prediction Simulator":

    st.title("🪐 Prediction Simulator")

    st.markdown("""
    Enter SDSS photometric magnitudes to classify astronomical objects.
    """)

    col1, col2 = st.columns(2)

    with col1:

        u = st.slider("u (Ultraviolet)", 10.0, 30.0, 18.0)
        g = st.slider("g (Green)", 10.0, 30.0, 17.0)
        r = st.slider("r (Red)", 10.0, 30.0, 16.0)

    with col2:

        i = st.slider("i (Near Infrared)", 10.0, 30.0, 15.0)
        z = st.slider("z (Infrared)", 10.0, 30.0, 14.0)

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    u_g = u - g
    g_r = g - r
    r_i = r - i
    i_z = i - z

    u_limit_flag = int(u > 24.5)
    r_limit_flag = int(r > 24.5)

    # PCA

    color_input = np.array([
        [u_g, g_r, r_i, i_z]
    ])

    color_scaled = pca_scaler.transform(color_input)

    pca_features = pca.transform(color_scaled)

    PC1 = pca_features[:, 0][0]
    PC2 = pca_features[:, 1][0]
    PC3 = pca_features[:, 2][0]

    # Stellar locus distance

    dist_locus = abs(
        g_r - locus_function(u_g)
    )

    # =====================================================
    # FINAL INPUT
    # =====================================================

    final_input = np.array([[
        r,
        u_g,
        g_r,
        r_i,
        i_z,
        PC1,
        PC2,
        PC3,
        dist_locus,
        u_limit_flag,
        r_limit_flag
    ]])

    scaled_input = scaler.transform(final_input)

    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(scaled_input)[0]

    probabilities = model.predict_proba(scaled_input)[0]

    predicted_label = encoder.inverse_transform([prediction])[0]

    confidence = np.max(probabilities) * 100

    # =====================================================
    # OUTPUT
    # =====================================================

    st.divider()

    st.subheader("🔮 Prediction Result")

    st.success(f"Predicted Class: {predicted_label}")

    st.metric("Confidence Score", f"{confidence:.2f}%")

    # =====================================================
    # PROBABILITY VISUALIZATION
    # =====================================================

    prob_df = pd.DataFrame({
        'Class': encoder.classes_,
        'Probability': probabilities
    })

    fig = px.bar(
        prob_df,
        x='Class',
        y='Probability',
        title='Prediction Probability Distribution'
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # SCIENTIFIC INTERPRETATION
    # =====================================================

    st.divider()

    st.subheader("📚 Astrophysical Interpretation")

    if predicted_label == "GALAXY":

        st.info("""
        The object exhibits photometric characteristics commonly associated
        with galaxies, especially broader red and infrared distributions.
        """)

    elif predicted_label == "STAR":

        st.info("""
        The object lies close to the stellar locus and resembles
        stellar photometric behavior.
        """)

    else:

        st.info("""
        The object exhibits unusual spectral energy distribution patterns
        consistent with quasar candidates.
        """)
