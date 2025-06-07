import os
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

# Define paths
dataset_dir = 'food-101/images'  # Path to Food-101 images
output_dir = 'food-101/processed_images'  # Where to save processed images
os.makedirs(output_dir, exist_ok=True)

# Define image transformation (resize and normalize for ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
])

# Function to process and save images
def preprocess_images(dataset_dir, output_dir):
    metadata = []
    
    # Loop through each category folder
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)
        
        # Process each image in the category
        for img_name in os.listdir(category_path):
            if not img_name.endswith('.jpg'):
                continue
                
            img_path = os.path.join(category_path, img_name)
            try:
                # Open and transform image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                
                # Convert back to PIL for saving
                img_pil = transforms.ToPILImage()(img_tensor)
                output_path = os.path.join(output_category_path, img_name)
                img_pil.save(output_path)
                
                # Add to metadata
                metadata.append({
                    'dish_name': category,
                    'image_path': output_path,
                    'is_veg': None,  # Placeholder for dietary tags
                    'is_spicy': None
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('food-101/dish_metadata.csv', index=False)
    print("Image preprocessing complete. Metadata saved to dish_metadata.csv")

# Run the preprocessing
if __name__ == "__main__":
    preprocess_images(dataset_dir, output_dir)