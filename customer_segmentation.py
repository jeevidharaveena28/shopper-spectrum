import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

rfm = pd.read_csv("data/rfm_data.csv")

print(rfm.head())

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

inertia = []

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(rfm_scaled)

    inertia.append(model.inertia_)


plt.figure(figsize=(8,5))

plt.plot(range(2,11), inertia, marker="o")

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.grid(True)

plt.show()

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(rfm_scaled)

    score = silhouette_score(rfm_scaled, labels)

    print(f"Clusters: {k}  Silhouette Score: {score:.3f}")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

print(rfm.head())

cluster_profile = rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()

print(cluster_profile)

segment_map = {
    0: "High-Value",
    1: "Regular",
    2: "Occasional",
    3: "At-Risk"
}

rfm["Segment"] = rfm["Cluster"].map(segment_map)


plt.figure(figsize=(10,6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Segment",
    palette="Set2"
)

plt.title("Customer Segments")

plt.show()

import pickle

pickle.dump(kmeans, open("models/kmeans_model.pkl", "wb"))

pickle.dump(scaler, open("models/scaler.pkl", "wb"))


import os
import pickle

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save KMeans model
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Save StandardScaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(" Models saved successfully!")