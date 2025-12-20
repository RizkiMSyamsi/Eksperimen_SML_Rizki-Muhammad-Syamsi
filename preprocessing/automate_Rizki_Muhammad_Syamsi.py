import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[col] >= lower) & (df[col] <= upper)]


def preprocess_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # ================= DATA CLEANING =================
    df = df.drop_duplicates()

    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]

    # ================= FEATURE REMOVAL =================
    df = df.drop(
        columns=[
            'TransactionNo',
            'Date',
            'ProductName',
            'CustomerNo'
        ],
        errors='ignore'
    )

    # ================= OUTLIER HANDLING =================
    for col in ['Price', 'Quantity']:
        df = remove_outliers_iqr(df, col)

    # ================= ENCODING =================
    df['Country'] = LabelEncoder().fit_transform(df['Country'])
    df['ProductNo'] = LabelEncoder().fit_transform(df['ProductNo'])

    # ================= SCALING =================
    scaler = StandardScaler()
    df[['Quantity']] = scaler.fit_transform(df[['Quantity']])

    # ================= FINAL FEATURE SELECTION =================
    df = df[['Price', 'Quantity', 'Country', 'ProductNo']]

    return df


if __name__ == "__main__":
    input_csv = "Sales-Transaction-v.4a.csv"
    output_csv = "preprocessing/Sales-Transaction-v.4a_preprocessing.csv"

    df_final = preprocess_data(input_csv)
    df_final.to_csv(output_csv, index=False)

    print(" Preprocessing selesai")
    print(" Input :", input_csv)
    print(" Output:", output_csv)
    print(" Kolom akhir:", list(df_final.columns))
