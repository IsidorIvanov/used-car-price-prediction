# used-car-price-prediction
ML regression project predicting used car prices listings (Germany, 1995–2023). Compares Ridge Regression, Random Forest, and XGBoost across MAE, RMSE, and R² metrics. Built with scikit-learn, XGBoost, pandas, and seaborn. 

## Simple preprocessing

The script `src/preprocessing.py` does a practical baseline clean-up:

- prints number of columns, column names, dtypes, and null counts
- converts numeric-like text columns to numeric values
- splits `registration_date` into `registration_month` and `registration_year`
- creates `car_age` from `year`
- drops low-value columns (`Unnamed: 0`, `offer_description`)
- removes duplicates and fills missing values (median for numeric, mode for categorical)

### Run

```powershell
python -m pip install -r requirements.txt
python src/preprocessing.py
```

### Output

Generated file:

- `DataSet/processed/processed_data.csv`

