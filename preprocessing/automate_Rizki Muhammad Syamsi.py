# automate_Rizki-Muhammad-Syamsi.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(input_path: str) -> pd.DataFrame:
    # ================= LOAD DATA =================
    df = pd.read_csv(input_path)

    # ================= CLEANING =================
    df = df.drop_duplicates()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]

    # ================= OUTLIER HANDLING (IQR) =================
    def remove_outliers_iqr(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return data[(data[col] >= lower) & (data[col] <= upper)]

    for col in ['Price', 'Quantity']:
        df = remove_outliers_iqr(df, col)

    # ================= FEATURE SELECTION =================
    # Fitur relevan untuk prediksi harga
    df = df[['Price', 'Quantity', 'Country', 'ProductNo']]

    # ================= ENCODING =================
    df['Country'] = LabelEncoder().fit_transform(df['Country'])
    df['ProductNo'] = LabelEncoder().fit_transform(df['ProductNo'])

    # ================= SCALING =================
    scaler = StandardScaler()
    df[['Quantity']] = scaler.fit_transform(df[['Quantity']])

    return df


# ================= MAIN =================
if __name__ == "__main__":
    input_csv = "Sales Transaction v.4a_preprocessing.csv"
    output_csv = "Sales Transaction v.4a.csv"

    df_final = preprocess_data(input_csv)
    df_final.to_csv(output_csv, index=False)

    print(" Preprocessing selesai")
    print(" Input :", input_csv)
    print(" Output:", output_csv)
    print(" Kolom akhir:", list(df_final.columns))
