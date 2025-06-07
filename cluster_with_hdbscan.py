import pandas as pd
import numpy as np
from tqdm import tqdm
import hdbscan
import umap
import os
import warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress FutureWarning for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Define paths
METADATA_FILE = 'food-101/dish_metadata.csv'
OUTPUT_METADATA_FILE = 'food-101/dish_metadata_with_clusters.csv'

# Load metadata
try:
    df = pd.read_csv(METADATA_FILE)
except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_FILE}")
    exit()

# Check for required column
if 'text_embedding_path' not in df.columns:
    print("Error: 'text_embedding_path' column not found in metadata.")
    exit()

# Use a subset for testing (adjust as needed)
SUBSET_SIZE = 10000  # Process first 10,000 rows
df = df.head(SUBSET_SIZE)
print(f"Using subset of {SUBSET_SIZE} embeddings for clustering.")

# Load text embeddings
text_embeddings = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading text embeddings"):
    try:
        embedding = np.load(row['text_embedding_path']).astype('float32')
        text_embeddings.append(embedding)
    except Exception as e:
        print(f"Error loading embedding for index {idx}: {e}")
        continue

if not text_embeddings:
    print("Error: No embeddings loaded. Check your embedding files.")
    exit()

# Convert to numpy array
text_embeddings = np.array(text_embeddings)
print(f"Loaded {text_embeddings.shape[0]} text embeddings with dimension {text_embeddings.shape[1]}")

# Optional: Reduce dimensionality with UMAP for faster clustering
print("Reducing dimensionality with UMAP...")
reducer = umap.UMAP(n_components=50, random_state=42)
text_embeddings_reduced = reducer.fit_transform(text_embeddings)
print(f"Reduced embeddings to dimension {text_embeddings_reduced.shape[1]}")

# Perform clustering with HDBSCAN
@ignore_warnings(category=ConvergenceWarning)
def run_hdbscan(embeddings):
    try:
        print("Starting HDBSCAN clustering...")
        min_cluster_size = 20  # Reduced for smaller subset
        min_samples = 5        # Controls clustering strictness
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
        cluster_labels = clusterer.fit_predict(embeddings)
        print("HDBSCAN clustering completed.")
        return cluster_labels
    except Exception as e:
        print(f"Error during clustering: {e}")
        return None

# Run clustering on reduced embeddings
cluster_labels = run_hdbscan(text_embeddings_reduced)
if cluster_labels is None:
    print("Clustering failed. Exiting.")
    exit()

# Add cluster labels to metadata
df['cluster_label'] = cluster_labels

# Save updated metadata
df.to_csv(OUTPUT_METADATA_FILE, index=False)
print(f"Clustering complete. Metadata updated at {OUTPUT_METADATA_FILE}")

# Analyze clusters
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of clusters: {n_clusters}")
cluster_sizes = pd.Series(cluster_labels).value_counts()
print("Cluster sizes:\n", cluster_sizes)

# Sample dishes from a few clusters (up to 3)
for cluster_id in range(min(3, n_clusters)):
    cluster_dishes = df[df['cluster_label'] == cluster_id]['cleaned_dish_name'].head(5).tolist()
    print(f"Sample dishes in cluster {cluster_id}:\n", cluster_dishes)
