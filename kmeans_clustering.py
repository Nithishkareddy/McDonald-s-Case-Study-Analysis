import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
file_path = "mcdonalds.csv"  # Change if needed
data = pd.read_csv(file_path)

# Convert categorical values to numerical
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

# Determine optimal clusters using the Elbow Method
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.savefig('elbow_method.png')
plt.show()

# Apply KMeans with chosen K (e.g., 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to data
data['Cluster'] = clusters

# Save clustered data
data.to_csv('segmented_data.csv', index=False)

# Scatter plot (using first two PCA components for visualization)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=clusters, palette="viridis")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.legend(title="Cluster")
plt.savefig('cluster_scatter.png')
plt.show()

# Cluster sizes
plt.figure(figsize=(6, 4))
sns.countplot(x=data["Cluster"], palette="coolwarm")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("Cluster Distribution")
plt.savefig('cluster_distribution.png')
plt.show()

print("Clustering completed. Results saved as 'segmented_data.csv'. Graphs saved as PNG files.")
