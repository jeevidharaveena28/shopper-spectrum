import pandas as pd

# Load dataset
df = pd.read_csv("data/online_retail.csv", encoding="ISO-8859-1")

# Display first 5 rows
print(df.head())

# Shape of dataset
print("\nShape:", df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Data information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())