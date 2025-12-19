import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_pipeline():
    # 1. Tentukan Path - Menggunakan nama file langsung agar aman di GitHub
    raw_path = 'train.csv'
    
    if not os.path.exists(raw_path):
        print(f"❌ Error: File {raw_path} tidak ditemukan di direktori saat ini.")
        return

    # 2. Load Data
    df = pd.read_csv(raw_path)

    # 3. Preprocessing (Handling Missing Values)
    df_clean = df.copy()
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    df_clean['LoanAmount'] = df_clean['LoanAmount'].fillna(df_clean['LoanAmount'].median())
    df_clean['Loan_Amount_Term'] = df_clean['Loan_Amount_Term'].fillna(df_clean['Loan_Amount_Term'].median())

    # 4. Cleaning & Encoding
    if 'Loan_ID' in df_clean.columns:
        df_clean.drop('Loan_ID', axis=1, inplace=True)
    df_clean['Dependents'] = df_clean['Dependents'].replace('3+', '3')

    # Label Encoding untuk Target
    le = LabelEncoder()
    df_clean['Loan_Status'] = le.fit_transform(df_clean['Loan_Status'])

    # One-Hot Encoding fitur kategorikal
    df_encoded = pd.get_dummies(df_clean, drop_first=True)

    # 5. Scaling
    X = df_encoded.drop('Loan_Status', axis=1)
    y = df_encoded['Loan_Status']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 6. GABUNGKAN & EXPORT
    df_final = X_scaled.copy()
    df_final['Loan_Status_Target'] = y.values

    # Simpan di direktori yang sama agar mudah dideteksi
    output_path = 'train_clean.csv'
    df_final.to_csv(output_path, index=False)

    print(f"✅ Otomatisasi Selesai!")
    print(f"✅ Data bersih 'train_clean.csv' telah dibuat.")

if __name__ == "__main__":
    run_pipeline()
