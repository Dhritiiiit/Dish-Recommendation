import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Define paths
METADATA_FILE = 'food-101/dish_metadata.csv'
EMBEDDINGS_DIR = 'food-101/image_embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)
# Remove the last layer (classification layer) to get embeddings
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()  # Set to evaluation mode

# Define image transformation for ResNet
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract embeddings
def extract_image_embeddings(metadata_file, embeddings_dir):
    # Load metadata
    df = pd.read_csv(metadata_file)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Process each image with a progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting image embeddings"):
        img_path = row['image_path']
        try:
            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
            
            # Get embedding
            with torch.no_grad():
                embedding = model(img_tensor).cpu().numpy().flatten()
            
            # Save embedding as .npy file
            embedding_path = os.path.join(embeddings_dir, f"{idx}.npy")
            np.save(embedding_path, embedding)
            
            # Add embedding path to metadata
            df.at[idx, 'image_embedding_path'] = embedding_path
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save updated metadata with embedding paths
    df.to_csv(metadata_file, index=False)
    print(f"Image embeddings extracted and saved. Metadata updated at {metadata_file}")

if __name__ == "__main__":
    extract_image_embeddings(METADATA_FILE, EMBEDDINGS_DIR)