import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df = pd.read_csv("data/cleaned_retail.csv")

print(df.head())

customer_product = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

print(customer_product.shape)

product_matrix = customer_product.T

print(product_matrix.shape)

similarity = cosine_similarity(product_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=product_matrix.index,
    columns=product_matrix.index
)

print(similarity_df.head())


def recommend_products(product_name, n=5):

    if product_name not in similarity_df.columns:
        return "Product not found."

    recommendations = (
        similarity_df[product_name]
        .sort_values(ascending=False)
        .iloc[1:n+1]
    )

    return recommendations

print(df["Description"].dropna().unique()[:20])

print(recommend_products("WHITE HANGING HEART T-LIGHT HOLDER"))

import pickle

with open("models/similarity_matrix.pkl", "wb") as f:
    pickle.dump(similarity_df, f)


product_names = similarity_df.index.tolist()

with open("models/product_names.pkl", "wb") as f:
    pickle.dump(product_names, f)