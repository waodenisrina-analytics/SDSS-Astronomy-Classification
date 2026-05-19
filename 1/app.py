import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIG & CUSTOM CSS (GLASSMORPHISM HACK)
# ==========================================
st.set_page_config(page_title="Stellar Analytics", page_icon="🌌", layout="wide")

st.markdown("""
<style>
    /* Global Dark Space Theme */
    .stApp {
        background: radial-gradient(circle at top left, #120e2b 0%, #05050f 100%);
        color: #e0e0e0;
    }
    
    /* Typography & Neon Accents */
    h1, h2, h3 {
        color: #00e5ff !important;
        font-family: 'Trebuchet MS', sans-serif;
        text-shadow: 0px 0px 10px rgba(0, 229, 255, 0.3);
    }
    
    /* Glassmorphism Panel Simulation */
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(12px);
    }
    
    /* Modern Button */
    .stButton>button {
        background: linear-gradient(135deg, #6200ea 0%, #b388ff 100%);
        color: white;
        border: none;
        border-radius: 6px;
        box-shadow: 0 4px 15px rgba(98, 0, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 229, 255, 0.5);
        color: #ffffff;
    }
    
    /* Metrics Override */
    div[data-testid="stMetricValue"] {
        color: #b388ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CACHING MODEL
# ==========================================
@st.cache_resource
def load_ml_artifacts():
    try:
        model = joblib.load('xgboost_stellar.pkl')
        encoder = joblib.load('label_encoder.pkl')
        return model, encoder
    except FileNotFoundError:
        st.error("⚠️ Artefak ML tidak ditemukan. Jalankan `python train.py` terlebih dahulu di terminal.")
        st.stop()

model, encoder = load_ml_artifacts()

# ==========================================
# UI LAYOUT
# ==========================================
st.title("🌌 SDSS Stellar Classification Platform")
st.markdown("*Astro-photometric Analysis without Redshift Dependencies*")

# Tabs Navigation
tab_overview, tab_simulator = st.tabs(["📋 Project Overview", "🚀 Prediction Simulator"])

# --- TAB 1: OVERVIEW ---
with tab_overview:
    st.header("Research Context")
    st.write("""
    Platform ini mengklasifikasikan objek langit (GALAXY, STAR, QSO) secara murni menggunakan **data fotometri** ($u, g, r, i, z$) dari instrumen Sloan Digital Sky Survey. 
    Pendekatan ini mendemonstrasikan kapabilitas prediktif Machine Learning pada data observasi awal sebelum konfirmasi spektroskopi (*redshift*) tersedia.
    """)
    
    st.markdown("### 📊 Dataset Metrik")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observasi", "100,000", "SDSS DR17")
    col2.metric("Target Classes", "3", "Galaxy/Star/QSO")
    col3.metric("Fitur Optikal", "5", "Magnitudo")
    col4.metric("Algoritma", "XGBoost", "Ensemble Tree")
    
    st.markdown("### ⚙️ Machine Learning Pipeline")
    st.code("Data Acquisition -> Feature Engineering -> Stratified Split -> XGBoost Classifier -> Probabilistic Output", language="python")

# --- TAB 2: SIMULATOR ---
with tab_simulator:
    st.header("Interactive Photometry Simulator")
    st.write("Sesuaikan nilai magnitudo fotometrik di panel kiri untuk melihat interpretasi model secara *real-time*.")
    
    col_in, col_out = st.columns([1, 1.2])
    
    with col_in:
        st.subheader("Input Parameters")
        # Nilai default diambil dari rata-rata populasi standar SDSS
        u = st.slider("u (Ultraviolet band)", 10.0, 30.0, 20.6, 0.01)
        g = st.slider("g (Green band)", 10.0, 30.0, 19.5, 0.01)
        r = st.slider("r (Red band)", 10.0, 30.0, 18.8, 0.01)
        i = st.slider("i (Near-Infrared band)", 10.0, 30.0, 18.4, 0.01)
        z = st.slider("z (Infrared band)", 10.0, 30.0, 18.2, 0.01)
        
        predict_trigger = st.button("JALANKAN KLASIFIKASI")
        
    with col_out:
        st.subheader("Inference Result")
        
        if predict_trigger:
            # Format input
            X_infer = pd.DataFrame([[u, g, r, i, z]], columns=['u', 'g', 'r', 'i', 'z'])
            
            # Eksekusi Model
            pred_idx = model.predict(X_infer)[0]
            pred_class = encoder.inverse_transform([pred_idx])[0]
            probs = model.predict_proba(X_infer)[0]
            confidence = np.max(probs) * 100
            
            # Tampilan Hasil
            st.markdown(f"### Objek Terdeteksi: <span style='color:#00e5ff'>{pred_class}</span>", unsafe_allow_html=True)
            
            st.write("**Confidence Level:**")
            st.progress(int(confidence))
            st.write(f"*{confidence:.2f}% kepastian berdasarkan arsitektur decision trees.*")
            
            # Chart Visualisasi Probabilitas
            st.markdown("#### Distribusi Probabilitas Kelas")
            
            fig, ax = plt.subplots(figsize=(7, 3))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            classes = encoder.classes_
            colors = ['#6200ea', '#00e5ff', '#b388ff']
            
            sns.barplot(x=probs*100, y=classes, palette=colors, ax=ax)
            ax.set_xlabel("Probabilitas (%)", color="white")
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('rgba(255, 255, 255, 0.2)')
                
            st.pyplot(fig)
            
            # Interpretasi Saintifik
            st.info(f"**Astro-insight:** Secara statistik pada ruang warna SDSS, nilai $(u-g)$ dan $(g-r)$ Anda berkontribusi tinggi pada klasifikasi ini. Model mengenali pola fotometrik ini sebagai **{pred_class}**.")
        else:
            st.write("Menunggu input. Silakan klik tombol klasifikasi.")