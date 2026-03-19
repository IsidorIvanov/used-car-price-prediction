import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR.parent / "DataSet" / "Raw" / "data.csv"
PROCESSED_DIR = BASE_DIR.parent / "DataSet" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "ready_for_use.csv"
def fix_numeric_column(series):
    fixed = (
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)   # european comma -> dot
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(fixed, errors="coerce")


def inspect_data(df):
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("Column names:")
    print(df.columns.tolist())
    print("Data types:")
    print(df.dtypes)
    print("Missing values:")
    print(df.isnull().sum())
    print("First 5 rows:")
    print(df.head())


def preprocess_data(df):
    data = df.copy()

    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    numeric_columns = [
        "year",
        "price_in_euro",
        "power_kw",
        "power_ps",
        "fuel_consumption_l_100km",
        "fuel_consumption_g_km",
        "mileage_in_km",
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = fix_numeric_column(data[col])

    if "registration_date" in data.columns:
        reg = pd.to_datetime(data["registration_date"], format="%m/%Y", errors="coerce")
        data["registration_month"] = reg.dt.month
        data["registration_year"] = reg.dt.year
        data = data.drop(columns=["registration_date"])

    if "year" in data.columns:
        current_year = datetime.now().year
        data["car_age"] = current_year - data["year"]
        data["car_age"] = data["car_age"].clip(lower=0)

    if "offer_description" in data.columns:
        data = data.drop(columns=["offer_description"])

    before = len(data)
    data = data.drop_duplicates()
    print(f"Removed {before - len(data)} duplicate rows")

    if "price_in_euro" in data.columns:
        data = data[data["price_in_euro"].notna()]
        data = data[data["price_in_euro"] > 0]

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
        else:
            mode_val = data[col].mode(dropna=True)
            if not mode_val.empty:
                data[col] = data[col].fillna(mode_val.iloc[0])
            else:
                data[col] = data[col].fillna("Unknown")

    return data


def main():
    print(f"Loading data from: {RAW_DATA_PATH}")
    raw_data = pd.read_csv(RAW_DATA_PATH)
    inspect_data(raw_data)

    # run preprocessing
    processed_data = preprocess_data(raw_data)

    # save to csv
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Saved to: {PROCESSED_DATA_PATH}")
    print(f"Rows: {len(processed_data)}")
    print(f"Columns: {len(processed_data.columns)}")
    print(f"Any nulls left: {processed_data.isnull().sum().sum()}")


if __name__ == "__main__":
    main()