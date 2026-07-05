import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_retail.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

print(snapshot_date)


rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print(rfm.head())

print(rfm.describe())

plt.figure(figsize=(8,5))
sns.histplot(rfm["Recency"], bins=30)
plt.title("Recency Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(rfm["Frequency"], bins=30)
plt.title("Frequency Distribution")
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(rfm["Monetary"], bins=30)
plt.title("Monetary Distribution")
plt.show()

rfm.to_csv("data/rfm_data.csv")