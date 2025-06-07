import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Define paths
METADATA_FILE = 'food-101/dish_metadata.csv'
EMBEDDINGS_DIR = 'food-101/text_embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set to evaluation mode

# Function to get BERT embedding for a single text
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=32)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding

# Function to extract text embeddings
def extract_text_embeddings(metadata_file, embeddings_dir):
    # Load metadata
    df = pd.read_csv(metadata_file)
    
    # Ensure 'cleaned_dish_name' exists
    if 'cleaned_dish_name' not in df.columns:
        raise ValueError("Column 'cleaned_dish_name' not found in metadata.")
    
    # Process each dish name with a progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting text embeddings"):
        dish_name = row['cleaned_dish_name']
        try:
            # Get embedding
            embedding = get_bert_embedding(dish_name)
            
            # Save embedding as .npy file
            embedding_path = os.path.join(embeddings_dir, f"{idx}.npy")
            np.save(embedding_path, embedding)
            
            # Add embedding path to metadata
            df.at[idx, 'text_embedding_path'] = embedding_path
            
        except Exception as e:
            print(f"Error processing {dish_name}: {e}")
    
    # Save updated metadata with embedding paths
    df.to_csv(metadata_file, index=False)
    print(f"Text embeddings extracted and saved. Metadata updated at {metadata_file}")

if __name__ == "__main__":
    extract_text_embeddings(METADATA_FILE, EMBEDDINGS_DIR)