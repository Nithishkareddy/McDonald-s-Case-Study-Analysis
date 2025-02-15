import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
file_path = "mcdonalds.csv"  # Ensure this is in the same directory
data = pd.read_csv(file_path)

# Convert categorical Yes/No columns to binary (0 = No, 1 = Yes)
binary_cols = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 
               'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
data[binary_cols] = data[binary_cols].applymap(lambda x: 1 if x == 'Yes' else 0)

# Encode categorical variables: 'Like', 'VisitFrequency', and 'Gender'
label_encoder = LabelEncoder()
data['Like'] = label_encoder.fit_transform(data['Like'])
data['VisitFrequency'] = label_encoder.fit_transform(data['VisitFrequency'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Select features for clustering
features = binary_cols + ['Like', 'Age', 'VisitFrequency', 'Gender']
X = data[features]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
num_clusters = 3  # Change as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Age'], y=data['Like'], hue=data['Cluster'], palette='viridis')
plt.title('Customer Segmentation Based on Age and Like Score')
plt.xlabel('Age')
plt.ylabel('Like Score')
plt.legend(title="Cluster")
plt.show()

# Save clustered dataset
data.to_csv("segmented_customers.csv", index=False)
print("Customer segmentation completed. Segmented data saved as 'segmented_customers.csv'.")
