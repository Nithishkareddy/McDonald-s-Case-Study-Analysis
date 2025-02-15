import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load your dataset (replace with the actual file path)
conv_df = pd.read_csv('mcdonalds.csv')  # Replace with the correct file path

# Check and print the dataset columns and datatypes
print(conv_df.dtypes)

# Separate categorical and numeric columns
categorical_cols = conv_df.select_dtypes(include=['object']).columns
numeric_cols = conv_df.select_dtypes(include=[np.number]).columns

# Apply Label Encoding for categorical columns (if you want to use label encoding)
label_encoder = LabelEncoder()
for col in categorical_cols:
    conv_df[col] = label_encoder.fit_transform(conv_df[col])

# Alternatively, you can use one-hot encoding for categorical columns:
# conv_df = pd.get_dummies(conv_df, drop_first=True)

# Now select only the numeric columns (after encoding)
conv_df_numeric = conv_df.select_dtypes(include=[np.number])

# Check if you have enough numeric features
print(conv_df_numeric.head())

# Standardize the data (numeric columns only)
scaler = StandardScaler()
conv_df_scaled = scaler.fit_transform(conv_df_numeric)

# Apply PCA (make sure you have more than one component)
pca = PCA(n_components=2)  # 2 components if you have more than 2 features
pca_result = pca.fit_transform(conv_df_scaled)

# Plotting the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], color='grey', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Analysis (Perceptual Map) of McDonald Fast Food Dataset Review')

# Plot the first two principal components as arrows (axes)
for i in range(pca.components_.shape[1]):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='red', alpha=0.5,
              head_width=0.05, head_length=0.1, width=0.005)

    # Label the arrows
    plt.text(pca.components_[0, i] * 1.15, pca.components_[1, i] * 1.15,
             conv_df_numeric.columns[i], color='r')

# Set plot limits
plt.xlim([-1.5, 1])
plt.ylim([-1, 1])

# Display grid
plt.grid(True)

# Show the plot
plt.show()
