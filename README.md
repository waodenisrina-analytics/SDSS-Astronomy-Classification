# 🌌 Klasifikasi Objek Langit Berdasarkan Data Fotometri SDSS

> Mengklasifikasikan Bintang, Galaksi, dan Quasar menggunakan Machine Learning berbasis data fotometri SDSS — tanpa memanfaatkan redshift.

**Tools:** Jupyter Notebook · Streamlit  
**Model:** XGBoost Classifier  
**Dataset:** [Stellar Classification Dataset - SDSS17 (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

🔗 **Live Dashboard:** [sdss-astronomy-classification.streamlit.app](https://sdss-astronomy-classification-f4raxkaqmckuav3hhw2jnt.streamlit.app/)

---

## 📌 Latar Belakang

Pemetaan alam semesta modern melalui **Sloan Digital Sky Survey (SDSS)** menghasilkan jutaan objek langit per malam. Secara tradisional, klasifikasi objek langit dilakukan melalui **analisis spektroskopi** untuk memperoleh nilai redshift — metode yang akurat namun mahal secara instrumentasi dan waktu observasi.

Proyek ini membuktikan bahwa **Machine Learning berbasis data fotometri** (pita u, g, r, i, z) dapat menggantikan peran spektroskopi awal dalam klasifikasi objek langit, sekaligus mengeliminasi risiko *data leakage* yang muncul bila redshift digunakan sebagai fitur prediktif.

---

## 🎯 Tujuan

1. Membangun model klasifikasi yang **independen terhadap redshift**, murni berbasis profil fotometri.
2. Melakukan **kuantifikasi Feature Importance** untuk membuktikan secara data-driven interaksi antar filter cahaya yang paling diskriminatif antar kelas.
3. Mengoptimalkan **sensitivitas terhadap kelas minoritas** (Quasar dan Bintang) agar model tidak bias pada kelas mayoritas (Galaksi).
4. Membandingkan performa **Logistic Regression, Random Forest, dan XGBoost** melalui metrik Recall dan F1-Score.

---

## 🗂️ Struktur Repositori

```
├── star_classification.csv       # Dataset mentah dari SDSS
├── Stellar_Classification.ipynb  # Notebook: EDA, Feature Engineering, Modelling
├── Penjelasan Proyek             # Penjelasan mulai dari latar belakang sampai pembahasan
├── train.py                      # Script training & export model artifacts
├── app.py                        # Dashboard Streamlit
├── model/
│   ├── xgb_model.pkl             # Model XGBoost terlatih
│   ├── scaler.pkl                # StandardScaler untuk fitur final
│   ├── pca_scaler.pkl            # StandardScaler untuk PCA
│   ├── pca.pkl                   # Objek PCA (3 komponen)
│   ├── label_encoder.pkl         # LabelEncoder kelas target
│   └── locus_coeffs.pkl          # Koefisien polinomial Stellar Locus
└── README.md
```

---

## 📊 Dataset

| Atribut | Detail |
|---|---|
| Sumber | Kaggle — fedesoriano (January 2022) |
| Ukuran | 100.000 baris observasi |
| Ukuran setelah cleaning | 99.999 baris |
| Variabel input | `u`, `g`, `r`, `i`, `z` (magnitudo fotometrik) |
| Variabel target | `class` → GALAXY, STAR, QSO |
| Distribusi kelas | GALAXY: 59.445 · STAR: 21.593 · QSO: 18.961 |

---

## ⚙️ Alur Pengerjaan

```
Data Fotometri SDSS
        ↓
1. Data Understanding & Cleaning
        ↓
2. Exploratory Data Analysis
        ↓
3. Rekayasa Fitur Astronomi
        ↓
4. Modelling & Evaluasi
        ↓
5. Deployment Dashboard Streamlit
```

---

## 🔬 Feature Engineering

Seluruh fitur dibangun atas dasar pengetahuan astronomi, bukan transformasi statistik semata.

| Fitur | Deskripsi |
|---|---|
| `r` | Magnitudo pita merah (red band) |
| `u-g` | Indeks warna UV–hijau, sensitif terhadap suhu bintang muda |
| `g-r` | Indeks warna hijau–merah, indikator suhu permukaan |
| `r-i` | Indeks warna merah–near-infrared, berkaitan dengan populasi bintang tua |
| `i-z` | Indeks warna near-infrared–infrared, sensitif terhadap debu antarbintang |
| `PC1`, `PC2`, `PC3` | Komponen PCA dari ruang warna (ortogonal, tidak berkorelasi) |
| `dist_locus` | Jarak absolut dari Stellar Locus pada diagram warna u-g vs g-r |
| `u_limit_flag` | Flag batas deteksi UV teleskop (u > 24.5 mag) |
| `r_limit_flag` | Flag batas deteksi merah teleskop (r > 24.5 mag) |

### Mengapa `dist_locus`?
Bintang normal mengikuti jalur melengkung sempit pada color-color diagram yang disebut **Stellar Locus**. Quasar dan Galaksi menyimpang jauh dari jalur ini akibat excess emisi UV dari cakram akresi. Fitur ini secara eksplisit mengkodekan informasi astrofisika tersebut ke dalam model.

---

## 🤖 Modelling & Evaluasi

Tiga model dilatih dan dibandingkan pada data yang sama:

| Model | F1 Macro | F1 Weighted | Recall Macro | Recall Weighted |
|---|---|---|---|---|
| Logistic Regression | 0.6706 | 0.7362 | 0.6724 | 0.7587 |
| Random Forest | 0.8511 | 0.8825 | 0.8414 | 0.8841 |
| **XGBoost** | **0.8541** | **0.8853** | **0.8458** | **0.8866** |

**XGBoost** dipilih sebagai model final karena konsisten unggul di seluruh metrik evaluasi.

### Konfigurasi XGBoost

```python
XGBClassifier(
    objective      = 'multi:softprob',
    num_class      = 3,
    n_estimators   = 300,
    max_depth      = 6,
    learning_rate  = 0.05,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    eval_metric    = 'mlogloss',
    random_state   = 42
)
```

### Feature Importance (XGBoost)

```
r-i          ██████████████████████  0.2051
g-r          █████████████████       0.1497
PC1          ████████████████        0.1326
PC2          ████████████            0.1072
u-g          ███████████             0.0982
i-z          ██████████              0.0860
r            █████████               0.0835
u_limit_flag █████                   0.0492
PC3          █████                   0.0443
dist_locus   ███                     0.0251
r_limit_flag ██                      0.0191
```

Indeks warna `r-i` dan `g-r` mendominasi karena paling diskriminatif dalam memisahkan distribusi energi spektral antar kelas objek langit.

---

## 🖥️ Menjalankan Proyek Secara Lokal

### 1. Clone repositori & install dependensi

```bash
git clone <url-repositori>
cd <nama-folder>
pip install -r requirements.txt
```

### 2. Latih ulang model

```bash
python train.py
```

Semua artifact model akan tersimpan di folder `model/`.

### 3. Jalankan dashboard

```bash
streamlit run app.py
```

---

## 📦 Dependensi Utama

```
pandas
numpy
scikit-learn
xgboost
streamlit
plotly
joblib
```

---

## 📚 Referensi

- fedesoriano. (January 2022). *Stellar Classification Dataset - SDSS17*. Kaggle.
- York, D. G., et al. (2000). The Sloan Digital Sky Survey: Technical Summary. *The Astronomical Journal*, 120(3), 1579.
- Ivezić, Ž., et al. (2019). *Statistics, Data Mining, and Machine Learning in Astronomy*. Princeton University Press.
