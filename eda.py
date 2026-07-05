import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Make plots look better
plt.style.use("ggplot")

df = pd.read_csv("data/cleaned_retail.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

country_sales = df["Country"].value_counts()

plt.figure(figsize=(12,6))
country_sales.plot(kind="bar")
plt.title("Transaction Volume by Country")
plt.xlabel("Country")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


top_products = (
    df.groupby("Description")["Quantity"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(12,6))
top_products.plot(kind="bar")
plt.title("Top 10 Selling Products")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

monthly_sales = df.groupby("Month")["TotalPrice"].sum()

plt.figure(figsize=(12,6))
monthly_sales.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["TotalPrice"], bins=50)
plt.title("Distribution of Transaction Amount")
plt.xlabel("Total Price")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x=df["UnitPrice"])
plt.title("Unit Price Distribution")
plt.show()


top_customers = (
    df.groupby("CustomerID")["TotalPrice"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10,5))
top_customers.plot(kind="bar")
plt.title("Top 10 Customers by Spending")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()