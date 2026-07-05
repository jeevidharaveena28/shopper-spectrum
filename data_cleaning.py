import pandas as pd

# Load dataset
df = pd.read_csv("data/online_retail.csv", encoding="ISO-8859-1")

print("Original Shape:", df.shape)

# Remove missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Remove invalid Quantity
df = df[df["Quantity"] > 0]

# Remove invalid UnitPrice
df = df[df["UnitPrice"] > 0]

# Create TotalPrice column
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("Cleaned Shape:", df.shape)

# Save cleaned data
df.to_csv("data/cleaned_retail.csv", index=False)

print("Data cleaned successfully!")