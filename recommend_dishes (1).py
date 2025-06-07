import pandas as pd
import numpy as np
import faiss
import os
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
import torch

# Define paths
METADATA_FILE = 'food-101/dish_metadata_with_clusters.csv'
IMAGE_INDEX_FILE = 'food-101/faiss_indexes/image_index.faiss'
TEXT_INDEX_FILE = 'food-101/faiss_indexes/text_index.faiss'

# Load metadata and FAISS indexes
try:
    df = pd.read_csv(METADATA_FILE)
    image_index = faiss.read_index(IMAGE_INDEX_FILE)
    text_index = faiss.read_index(TEXT_INDEX_FILE)
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    exit()

# Set device to CPU for M3 compatibility
device = torch.device('cpu')
print(f"Using device: {device}")

# Load pre-trained models for query embedding
# ResNet for images
resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')  # Updated for newer torchvision
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
resnet.eval()
resnet.to(device)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# BERT for text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()
bert_model.to(device)

# Function to get image embedding
def get_image_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = image_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(img_tensor).cpu().numpy().flatten()
        return embedding.astype('float32')
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to get text embedding
def get_text_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=32).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return embedding.astype('float32')
    except Exception as e:
        print(f"Error processing text '{text}': {e}")
        return None

# Function to recommend dishes
def recommend_dishes(query_type, query_input, k=5):
    if query_type == 'image':
        query_embedding = get_image_embedding(query_input)
        index = image_index
    else:  # text
        query_embedding = get_text_embedding(query_input)
        index = text_index
    
    if query_embedding is None:
        print(f"Failed to generate embedding for {query_type} query.")
        return []
    
    # Search FAISS index
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    
    # Get recommendations
    recommendations = df.iloc[indices[0]][['cleaned_dish_name', 'image_path', 'cluster_label']].to_dict('records')
    return recommendations

# Example usage
if __name__ == "__main__":
    # Test with an image query
    test_image = 'food-101/processed_images/pizza/1001116.jpg'  # Replace with a valid image path
    if os.path.exists(test_image):
        image_recommendations = recommend_dishes('image', test_image)
        print("Image-based recommendations:")
        for rec in image_recommendations:
            print(rec)
    else:
        print(f"Test image not found: {test_image}")
    
    # Test with a text query
    test_text = "pizza"
    text_recommendations = recommend_dishes('text', test_text)
    print("Text-based recommendations:")
    for rec in text_recommendations:
        print(rec)
