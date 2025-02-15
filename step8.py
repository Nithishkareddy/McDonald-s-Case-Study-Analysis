import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("mcdonalds.csv")

# Print column names and sample data
print("Column Names:", df.columns)
print("\nFirst few rows:")
print(df.head())
print("\nUnique values in Like column:")
print(df["Like"].unique())
print("\nUnique values in VisitFrequency column:")
print(df["VisitFrequency"].unique())

# Print missing value counts
print("\nMissing values per column:")
print(df.isnull().sum())

# Get attribute columns
attribute_cols = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 
                 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']

# Convert Yes/No columns to 1/0
for col in attribute_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Handle special cases in Like column
def extract_number_from_like(value):
    if pd.isna(value):
        return np.nan
    
    # If it's already a number, return it
    if isinstance(value, (int, float)):
        return value
    
    # Try to extract numbers from strings like "+2", "-3", "I love it!5"
    matches = re.findall(r'[+-]?\d+', str(value))
    if matches:
        return int(matches[0])
    else:
        # Handle cases with no numbers
        if "love" in str(value).lower():
            return 5  # Assuming "I love it" is the highest rating
        if "hate" in str(value).lower():
            return -5  # Assuming "I hate it" is the lowest rating
        return np.nan  # If we can't extract a meaningful value

# Apply the function to convert Like values
df["Like"] = df["Like"].apply(extract_number_from_like)

# Convert VisitFrequency to numeric scale
visit_freq_map = {
    "Never": 1,
    "Every three months": 2,
    "Once a month": 3,
    "Once a week": 4,
    "More often": 5
}
df["VisitFrequency"] = df["VisitFrequency"].map(visit_freq_map)

# Verify conversions worked
print("\nAfter conversions - first few rows:")
print(df.head())
print("\nUnique values in Like column after conversion:")
print(df["Like"].unique())
print("\nUnique values in VisitFrequency column after conversion:")
print(df["VisitFrequency"].unique())

# Check if we have any valid rows for essential columns
valid_rows = df["Like"].notna() & df["VisitFrequency"].notna() & df["Gender"].notna()
print(f"\nNumber of valid rows for essential columns: {valid_rows.sum()} out of {len(df)}")

if valid_rows.sum() == 0:
    raise ValueError("No valid data rows found with non-missing values in essential columns")

# Filter to only keep valid rows
df = df[valid_rows].copy()

# Prepare data for clustering
X = df[attribute_cols].copy()

# Impute any remaining missing values in attribute columns
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Perform K-means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['k4'] = kmeans.fit_predict(X_scaled) + 1  # Adding 1 to make segments 1-4 instead of 0-3

# Compute mean VisitFrequency for each segment
visit = df.groupby("k4")["VisitFrequency"].mean()
print("\nVisit Frequency per Segment:\n", visit)

# Compute mean Liking McDonald's value for each segment
like = df.groupby("k4")["Like"].mean()
print("\nLiking McDonald's per Segment:\n", like)

# Convert 'Gender' to numeric (1 for Female, 0 for Male)
df["Female"] = (df["Gender"] == "Female").astype(int)

# Compute percentage of Female Consumers per Segment
female = df.groupby("k4")["Female"].mean()
print("\nPercentage of Female Consumers per Segment:\n", female)

# Plot the Segment Evaluation Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(visit, like, s=1000 * female, alpha=0.6, edgecolors="black")
plt.xlabel("Visit Frequency")
plt.ylabel("Liking McDonald's")

# Set the axis limits as in the original code
plt.xlim(2, 4.5)
plt.ylim(-3, 3)

# Label each segment
for i, (x, y) in enumerate(zip(visit, like)):
    plt.text(x, y, str(i+1), fontsize=12, ha="center", va="center")

plt.title("Segment Evaluation Plot")
plt.grid(True)

# Add a legend for the bubble size
sizes = [0.2, 0.5, 0.8]
labels = ["20% female", "50% female", "80% female"]
legend_bubbles = []
for size in sizes:
    legend_bubbles.append(plt.scatter([], [], s=1000 * size, alpha=0.6, 
                                      edgecolors="black", color=scatter.get_facecolors()[0]))
plt.legend(legend_bubbles, labels, title="Gender Composition", 
           loc="upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig("segment_evaluation_plot.png")  # Save the plot as an image file
plt.show()

# Print cluster centers to understand what each segment represents
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_profile = pd.DataFrame(
    cluster_centers,
    columns=attribute_cols,
    index=[f"Segment {i+1}" for i in range(4)]
)
print("\nCluster Centers (Original Scale):")
print(cluster_profile)

# Print segment sizes
segment_sizes = df['k4'].value_counts().sort_index()
print("\nSegment Sizes:")
print(segment_sizes)

# Print interpretation of each segment
print("\nSegment Interpretation:")
for i in range(4):
    segment_num = i + 1
    visit_val = visit[segment_num]
    like_val = like[segment_num]
    female_pct = female[segment_num] * 100
    size = segment_sizes[segment_num]
    
    # Interpret visit frequency
    if visit_val < 2.5:
        visit_desc = "infrequent visitors"
    elif visit_val < 3.5:
        visit_desc = "moderately frequent visitors"
    else:
        visit_desc = "frequent visitors"
    
    # Interpret liking
    if like_val < -1:
        like_desc = "strongly dislike"
    elif like_val < 0:
        like_desc = "somewhat dislike"
    elif like_val < 1:
        like_desc = "neutral about"
    elif like_val < 2:
        like_desc = "somewhat like"
    else:
        like_desc = "strongly like"
    
    # Interpret gender composition
    if female_pct > 60:
        gender_desc = "predominantly female"
    elif female_pct < 40:
        gender_desc = "predominantly male"
    else:
        gender_desc = "gender-balanced"
    
    print(f"Segment {segment_num}: {size} customers who are {visit_desc} that {like_desc} McDonald's, {gender_desc} ({female_pct:.1f}% female)")