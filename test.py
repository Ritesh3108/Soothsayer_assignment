import pandas as pd
import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load Data
transaction_01 = pd.read_csv(r"Transactional_data_retail_01.csv")
transaction_02 = pd.read_csv(r"Transactional_data_retail_02.csv")
customer_data = pd.read_csv(r"CustomerDemographics.csv")
product_info = pd.read_csv(r"ProductInfo.csv")

#Merge Data
merged_data = pd.merge(transaction_01, transaction_02, how='outer')
merged_data = pd.merge(merged_data, customer_data, on='CustomerID', how='left')
merged_data = pd.merge(merged_data, product_info, on='StockCode', how='left')


# Date Formatting with error handling
merged_data['InvoiceDate'] = pd.to_datetime(merged_data['InvoiceDate'], errors='coerce')

# Drop rows where 'TransactionDate' cannot be parsed
merged_data.dropna(subset=['InvoiceDate'], inplace=True)

# Handling missing values based on column type
for col in merged_data.columns:
    if merged_data[col].dtype == 'float64' or merged_data[col].dtype == 'int64':
        # For numerical columns, fill missing values with the median
        median_value = merged_data[col].median()
        merged_data[col].fillna(median_value, inplace=True)
    elif merged_data[col].dtype == 'object':
        # For categorical columns, fill with the most frequent value
        mode_value = merged_data[col].mode()[0]
        merged_data[col].fillna(mode_value, inplace=True)
merged_data['InvoiceDate'].fillna(method='ffill', inplace=True)


# SQL-style querying


# Creating a connection to SQLite (or SQL server)
conn = sqlite3.connect('data.db')
merged_data.to_sql('retail_data', conn, if_exists='replace')

# Query: Find total sales per product
query = """
SELECT StockCode, SUM(Quantity) as TotalSales
FROM retail_data
GROUP BY StockCode
ORDER BY TotalSales DESC
LIMIT 10;
"""
top_10_products = pd.read_sql(query, conn)

# Calculate revenue (Price * Quantity) and sort by revenue
merged_data['Revenue'] = merged_data['Price'] * merged_data['Quantity']

# Top 10 products by revenue
top_10_revenue_products = merged_data.groupby('StockCode')['Revenue'].sum().sort_values(ascending=False).head(10)


# Assuming 'merged_data' contains your time series data with 'InvoiceDate' and 'Quantity'
# Filter the data for a specific product or the entire time series
product_data = merged_data[['InvoiceDate', 'Quantity']].set_index('InvoiceDate')

# Resample the data by week to capture trends and seasonality
product_data_resampled = product_data.resample('W').sum()

# Step 4.1: Plot ACF and PACF to Identify Trends, Seasonality, and Lags
plt.figure(figsize=(12, 6))

# ACF Plot
plt.subplot(121)
plot_acf(product_data_resampled['Quantity'], lags=40, ax=plt.gca())
plt.title("ACF Plot")

# PACF Plot
plt.subplot(122)
plot_pacf(product_data_resampled['Quantity'], lags=40, ax=plt.gca())
plt.title("PACF Plot")

plt.tight_layout()
plt.show()

p = 2  # From PACF plot (sharp drop-off point)
d = 1  # Order of differencing (for stationarity)
q = 2  # From ACF plot (sharp drop-off point)

#  Fit the ARIMA Model based on ACF and PACF
model = ARIMA(product_data_resampled['Quantity'], order=(p, d, q))
model_fit = model.fit()

#  Forecast for the next 15 weeks
forecast = model_fit.forecast(steps=15)

#  Plot forecast vs historical data
plt.plot(product_data_resampled.index, product_data_resampled['Quantity'], label='Historical Demand')
plt.plot(forecast.index, forecast, label='Forecasted Demand', color='red')
plt.title("Demand Forecast for the Next 15 Weeks")
plt.legend()
plt.show()



X = merged_data[['StockCode', 'Price', 'Quantity']]
y = merged_data['Revenue']

# Label encode 'StockCode' as it is a categorical variable
label_encoder = LabelEncoder()
X['StockCode'] = label_encoder.fit_transform(X['StockCode'])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)

# Evaluation metrics
rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)

print(f"RMSE: {rmse}, MAE: {mae}")


from sklearn.preprocessing import LabelEncoder

# Label encode 'StockCode' to avoid high cardinality issue
label_encoder = LabelEncoder()
X['StockCode'] = label_encoder.fit_transform(X['StockCode'])

# Proceed with Train/Test split and model fitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)

# Evaluation metrics
rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)

print(f"RMSE: {rmse}, MAE: {mae}")
 


# Assuming 'merged_data' contains the preprocessed data and 'InvoiceDate' is present
# Features including 'StockCode' which needs to be label encoded
X = merged_data[['InvoiceDate', 'StockCode', 'Price', 'Quantity']]  # Features
y = merged_data['Revenue']  # Target

# Label encode 'StockCode' (as it's categorical)
label_encoder = LabelEncoder()
X['StockCode'] = label_encoder.fit_transform(X['StockCode'])

# Sort data by 'InvoiceDate'
X = X.sort_values(by='InvoiceDate')
y = y.loc[X.index]  # Align the target with sorted 'X'

# Drop 'InvoiceDate' from X since it's not a feature
X = X.drop(columns=['InvoiceDate'], errors='ignore')

# Time-based cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Initialize the XGBoost model
model = XGBRegressor()

# Lists to store performance metrics
rmse_scores = []
mae_scores = []

# Time-based cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Evaluate RMSE and MAE
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    
    # Store results
    rmse_scores.append(rmse)
    mae_scores.append(mae)

# Step 7: Error and Evaluation Metrics
# Output the cross-validated RMSE and MAE scores
print(f"Cross-validated RMSE scores: {rmse_scores}")
print(f"Cross-validated MAE scores: {mae_scores}")
print(f"Average RMSE: {sum(rmse_scores) / len(rmse_scores)}")
print(f"Average MAE: {sum(mae_scores) / len(mae_scores)}")

