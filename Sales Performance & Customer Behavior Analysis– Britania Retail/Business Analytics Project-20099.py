import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ------------------- Connect to MySQL -------------------
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="pitbulldog@7",
    database="britania",  # correct DB name
    use_pure=True
)

# ------------------- Load Data -------------------
query = "SELECT * FROM `britania retail sales`"

df = pd.read_sql(query, con=mydb)
mydb.close()

# ------------------- Inspect Data -------------------
print("Columns in the dataset:", df.columns)
print(df.head())
print(df.info())
print(df.describe(include="all"))

# ------------------- Preprocess Data -------------------
# Handle missing values
df.fillna(0, inplace=True)  # or you can use df.fillna(df.mean()) for numeric

# Convert 'Month' to datetime
if 'Month' in df.columns:
    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    df = df.sort_values('Month')
else:
    print("Column 'Month' not found!")

# ------------------- Visualizations -------------------
# Sales Revenue Over Time
if 'Month' in df.columns and 'Sales_Revenue' in df.columns:
    plt.figure(figsize=(12,6))
    plt.plot(df['Month'], df['Sales_Revenue'])
    plt.title("Sales Revenue Over Time")
    plt.xlabel("Month")
    plt.ylabel("Sales Revenue")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Marketing Spend vs Sales Revenue
if 'Marketing_Spend' in df.columns and 'Sales_Revenue' in df.columns:
    plt.figure(figsize=(8,6))
    plt.scatter(df['Marketing_Spend'], df['Sales_Revenue'])
    plt.title("Marketing Spend vs Sales Revenue")
    plt.xlabel("Marketing Spend")
    plt.ylabel("Sales Revenue")
    plt.show()

# Histogram of Store Visits
if 'Store Visits' in df.columns:
    plt.figure(figsize=(8,5))
    plt.hist(df['Store Visits'], bins=20)
    plt.title("Histogram of Store Visits")
    plt.xlabel("Store Visits")
    plt.ylabel("Frequency")
    plt.show()

# Correlation Heatmap (numeric columns only)
num_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(8,6))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ------------------- Linear Regression -------------------
# Select features and target
features = ['Store Visits', 'Marketing_Spend', 'Discount_Percentage', 'Competitor_Price_Index']
target = 'Sales_Revenue'

# Check if all required columns exist
missing_cols = [col for col in features+[target] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

# Print regression coefficients
print("\nRegression Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.2f}")

# Predicted vs Actual Sales Revenue & Residual Plot
plt.figure(figsize=(12,5))

# Predicted vs Actual
plt.subplot(1, 2, 1)
plt.scatter(y, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'red', lw=2)
plt.xlabel('Actual Sales Revenue')
plt.ylabel('Predicted Sales Revenue')
plt.title('Predicted vs Actual')
plt.grid(True)

# Residual Plot
plt.subplot(1, 2, 2)
residuals = y - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Sales Revenue')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------- Forecast for Next Month -------------------
last_row = df.iloc[-1]
next_month_input = [[
    last_row['Store Visits'],
    last_row['Marketing_Spend'] * 1.10,  # assume 10% increase in marketing
    last_row['Discount_Percentage'],
    last_row['Competitor_Price_Index']
]]

# Scale and predict
next_month_scaled = scaler.transform(next_month_input)
forecast = model.predict(next_month_scaled)[0]
print(f"\nForecasted Sales Revenue for Next Month: ${forecast:.2f}")
