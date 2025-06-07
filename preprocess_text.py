import pandas as pd
import os

# Define paths
CLASSES_FILE = 'food-101/meta/classes.txt'
METADATA_FILE = 'food-101/dish_metadata.csv'

# Load and clean dish names
def load_and_clean_dish_names(classes_file):
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Classes file not found at {classes_file}")
    
    with open(classes_file, 'r') as f:
        dish_names = [line.strip() for line in f]
    
    # Clean names: replace underscores with spaces
    cleaned_names = [name.replace('_', ' ') for name in dish_names]
    
    # Create a mapping
    name_mapping = {original: cleaned for original, cleaned in zip(dish_names, cleaned_names)}
    return name_mapping

# Update metadata with cleaned names
def update_metadata(classes_file, metadata_file):
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
    
    # Load metadata
    df = pd.read_csv(metadata_file)
    
    # Clean dish names
    name_mapping = load_and_clean_dish_names(classes_file)
    df['cleaned_dish_name'] = df['dish_name'].map(name_mapping)
    
    # Save updated metadata
    df.to_csv(metadata_file, index=False)
    print(f"Text preprocessing complete. Metadata updated at {metadata_file}")

if __name__ == "__main__":
    update_metadata(CLASSES_FILE, METADATA_FILE)
