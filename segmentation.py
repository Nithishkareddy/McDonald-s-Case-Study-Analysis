import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "mcdonalds.csv"  # Ensure this is the correct path
df = pd.read_csv(file_path)

# Display basic info
print(df.info())
print(df.head())

# Handling non-numeric values in "Like" column
df['Like'] = df['Like'].astype(str).str.extract('(\d+)')  # Extract numbers
df['Like'] = pd.to_numeric(df['Like'], errors='coerce')  # Convert to numeric
df.dropna(subset=['Like'], inplace=True)  # Remove NaN values

# Normalize numeric features for clustering
features = df.select_dtypes(include=[np.number])  # Select only numeric columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform KMeans clustering
num_clusters = 3  # Adjust based on business logic
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(scaled_features)

# Visualizing clusters
plt.figure(figsize=(8, 6))
sns.boxplot(x='Segment', y='Like', data=df)
plt.title("Customer Segments Based on Likes")
plt.xlabel("Segment")
plt.ylabel("Like Score")
plt.show()

# Save segmented dataset
df.to_csv("segmented_customers.csv", index=False)

print("Market segmentation completed successfully. Segmented data saved as 'segmented_customers.csv'.")
