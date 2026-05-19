import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

def main():
    print("1. Memuat dataset SDSS...")
    df = pd.read_csv('star_classification.csv')
    
    # Fokus murni pada fitur fotometri tanpa redshift
    features = ['u', 'g', 'r', 'i', 'z']
    X = df[features]
    y = df['class']
    
    print("2. Preprocessing Data...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Stratified split penting untuk menangani class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("3. Melatih Model XGBoost...")
    # XGBoost memiliki performa tinggi untuk tabular data
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"4. Evaluasi Selesai. Akurasi Test: {accuracy:.4f}")
    
    print("5. Mengekspor artefak model...")
    joblib.dump(model, 'xgboost_stellar.pkl')
    joblib.dump(encoder, 'label_encoder.pkl')
    print("Berhasil! Artefak siap digunakan oleh dashboard.")

if __name__ == '__main__':
    main()