import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("mcdonalds.csv")

# Preview dataset
print(df.head())

# Convert categorical variables to numerical (if necessary)
df_encoded = pd.get_dummies(df)

# Normalize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Step 1: Apply K-Means Clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(df_scaled)

# Step 2: Perform Hierarchical Clustering on Attributes
attribute_dist = linkage(df_scaled.T, method="ward")

# Step 3: Create a Segment Profile Plot
plt.figure(figsize=(12, 6))
dendrogram(attribute_dist, labels=df_encoded.columns, orientation="right")
plt.title("Hierarchical Clustering of Attributes")
plt.xlabel("Distance")
plt.ylabel("Attributes")
plt.show()

# Step 4: Create a Segment Distribution Plot
plt.figure(figsize=(8, 4))
sns.countplot(x=df['Segment'], palette="viridis")
plt.title("Distribution of Segments")
plt.xlabel("Segment")
plt.ylabel("Count")
plt.show()

# Step 5: Principal Component Analysis (PCA) for Segment Separation
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(df_pca, columns=["PC1", "PC2"])
df_pca['Segment'] = df['Segment']

# Step 6: Plot Segment Separation using PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC2", hue=df_pca['Segment'], palette="Set1", data=df_pca, alpha=0.7)
plt.title("Segment Separation Plot (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Segments")
plt.show()
