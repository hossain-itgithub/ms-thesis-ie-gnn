import os
import shutil

def create_new_dataset(base_dir, new_dataset_dir):
    entities_dir = os.path.join(base_dir, 'entities')
    images_dir = os.path.join(base_dir, 'images')
    box_dir = os.path.join(base_dir, 'box')
    
    new_images_dir = os.path.join(new_dataset_dir, 'images')
    new_box_dir = os.path.join(new_dataset_dir, 'box')
    new_entities_dir = os.path.join(new_dataset_dir, 'entities')
    
    # Create new directories
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_box_dir, exist_ok=True)
    os.makedirs(new_entities_dir, exist_ok=True)
    
    # List all entity files
    entity_files = os.listdir(entities_dir)
    
    for entity_file in entity_files:
        # Define file paths
        source_entity_path = os.path.join(entities_dir, entity_file)
        dest_entity_path = os.path.join(new_entities_dir, entity_file)
        
        # Copy entity file
        shutil.copy2(source_entity_path, dest_entity_path)
        
        # Extract file name without extension
        base_filename = os.path.splitext(entity_file)[0]
        
        # Define corresponding image and box file paths
        source_image_path = os.path.join(images_dir, base_filename + '.jpg')  # Assuming images are .jpg files
        source_box_path = os.path.join(box_dir, base_filename + '.txt')  # Assuming box files are .txt files
        
        # Define destination paths
        dest_image_path = os.path.join(new_images_dir, base_filename + '.jpg')
        dest_box_path = os.path.join(new_box_dir, base_filename + '.txt')
        
        # Copy image file if exists
        if os.path.exists(source_image_path):
            shutil.copy2(source_image_path, dest_image_path)
        
        # Copy box file if exists
        if os.path.exists(source_box_path):
            shutil.copy2(source_box_path, dest_box_path)

    print("New dataset created successfully!")

# Example usage
base_dir = 'origDataset'
new_dataset_dir = 'ArabTrainDataset'

create_new_dataset(base_dir, new_dataset_dir)
