import streamlit as st
import pandas as pd
import pickle

from sklearn.metrics.pairwise import cosine_similarity

st.sidebar.title(" Shopper Spectrum")
st.sidebar.markdown("---")

st.title(" Shopper Spectrum")

st.markdown("""
### Customer Segmentation & Product Recommendation System

This application helps businesses:
-  Recommend similar products
-  Segment customers using RFM Analysis and KMeans Clustering
""")

# Load models
# Load cleaned dataset
df = pd.read_csv("data/cleaned_retail.csv")

# Create customer-product matrix
customer_product = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

# Create product matrix
product_matrix = customer_product.T

# Calculate similarity matrix
similarity = cosine_similarity(product_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=product_matrix.index,
    columns=product_matrix.index
)
product_names = pickle.load(open("models/product_names.pkl", "rb"))
kmeans = pickle.load(open("models/kmeans_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒")

st.title(" Shopper Spectrum")
st.write("Customer Segmentation & Product Recommendation System")

menu = st.sidebar.selectbox(
    "Choose Module",
    ["Product Recommendation", "Customer Segmentation"]
)

# --------------------------
# Product Recommendation
# --------------------------

if menu == "Product Recommendation":

    st.header("Product Recommendation")

    product = st.selectbox(
        "Select Product",
        product_names
    )

    if st.button("Get Recommendations"):

        recommendations = (
            similarity_df[product]
            .sort_values(ascending=False)
            .iloc[1:6]
        )

        st.success("Recommended Products")

        for item in recommendations.index:
            st.write("✅", item)

# --------------------------
# Customer Segmentation
# --------------------------

else:

    st.header("Customer Segmentation")

    recency = st.number_input("Recency", min_value=0)

    frequency = st.number_input("Frequency", min_value=0)

    monetary = st.number_input("Monetary", min_value=0.0)

    if st.button("Predict Segment"):

        values = scaler.transform([[recency, frequency, monetary]])

        cluster = kmeans.predict(values)[0]

        segment_map = {
            0: "High-Value",
            1: "Regular",
            2: "Occasional",
            3: "At-Risk"
        }

        st.success(f"Customer Segment: {segment_map[cluster]}")



