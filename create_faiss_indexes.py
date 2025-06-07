import pandas as pd
import numpy as np
import faiss
import os
from tqdm import tqdm

# Define paths
METADATA_FILE = 'food-101/dish_metadata.csv'
IMAGE_EMBEDDINGS_DIR = 'food-101/image_embeddings'
TEXT_EMBEDDINGS_DIR = 'food-101/text_embeddings'
INDEX_DIR = 'food-101/faiss_indexes'
os.makedirs(INDEX_DIR, exist_ok=True)

# Paths to save FAISS indexes
IMAGE_INDEX_PATH = os.path.join(INDEX_DIR, 'image_index.faiss')
TEXT_INDEX_PATH = os.path.join(INDEX_DIR, 'text_index.faiss')

# Function to create FAISS index
def create_faiss_index(embeddings, dimension):
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(embeddings)  # Add embeddings to index
    return index

# Function to index embeddings
def index_embeddings(metadata_file, image_embeddings_dir, text_embeddings_dir, image_index_path, text_index_path):
    # Load metadata
    df = pd.read_csv(metadata_file)
    
    # Check required columns
    required_columns = ['image_embedding_path', 'text_embedding_path']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns in metadata: {}".format(required_columns))
    
    # Initialize lists to store embeddings
    image_embeddings = []
    text_embeddings = []
    
    # Load embeddings with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading embeddings"):
        try:
            # Load image embedding
            image_embedding = np.load(row['image_embedding_path']).astype('float32')
            image_embeddings.append(image_embedding)
            
            # Load text embedding
            text_embedding = np.load(row['text_embedding_path']).astype('float32')
            text_embeddings.append(text_embedding)
            
        except Exception as e:
            print(f"Error loading embeddings for index {idx}: {e}")
            continue
    
    # Convert to numpy arrays
    image_embeddings = np.array(image_embeddings)
    text_embeddings = np.array(text_embeddings)
    
    # Create and save image index
    image_index = create_faiss_index(image_embeddings, dimension=image_embeddings.shape[1])
    faiss.write_index(image_index, image_index_path)
    print(f"Image FAISS index saved to {image_index_path}")
    
    # Create and save text index
    text_index = create_faiss_index(text_embeddings, dimension=text_embeddings.shape[1])
    faiss.write_index(text_index, text_index_path)
    print(f"Text FAISS index saved to {text_index_path}")

if __name__ == "__main__":
    index_embeddings(METADATA_FILE, IMAGE_EMBEDDINGS_DIR, TEXT_EMBEDDINGS_DIR, IMAGE_INDEX_PATH, TEXT_INDEX_PATH)