import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv("mcdonalds.csv")  # Ensure the file is in your working directory

# 2. Convert categorical 'YES'/'NO' to binary (1/0) for the perception variables
perception_cols = df.columns[:11]  # Assuming first 11 columns are perception variables
df_binary = df[perception_cols].applymap(lambda x: 1 if x == 'YES' else 0)

# 3. Calculate mean perception scores
mean_scores = df_binary.mean()
print("Mean Perception Scores:\n", mean_scores)

# 4. Standardize the data for PCA
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_binary)

# 5. Perform PCA
pca = PCA(n_components=2)  # First 2 Principal Components
pca_result = pca.fit_transform(df_scaled)

# 6. Create a DataFrame for visualization
df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

# 7. Plot the Perceptual Map
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_pca["PC1"], y=df_pca["PC2"], alpha=0.5)
plt.axhline(0, linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')
plt.title("Perceptual Map of McDonald's Consumer Perceptions")
plt.xlabel("PC1 (Perception Dimension)")
plt.ylabel("PC2 (Price Dimension)")
# Annotate attributes (original features)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
for i, feature in enumerate(perception_cols):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.7)
    plt.text(loadings[i, 0], loadings[i, 1], feature, fontsize=12, color='black')

plt.show()
