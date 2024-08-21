import os
import shutil
import random

def create_directory_structure(base_path):
    if not os.path.exists(base_path):
        os.makedirs(os.path.join(base_path, 'images'))
        os.makedirs(os.path.join(base_path, 'box'))
        os.makedirs(os.path.join(base_path, 'entities'))

def copy_files(file_list, src_dir, dest_dir):
    for file in file_list:
        for sub_dir in ['images', 'box', 'entities']:
            src_path = os.path.join(src_dir, sub_dir, file + ('.jpg' if sub_dir == 'images' else '.txt'))
            dest_path = os.path.join(dest_dir, sub_dir, file + ('.jpg' if sub_dir == 'images' else '.txt'))
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)

def main(dataset_dir, train_dir, test_dir, train_size):
    images_dir = os.path.join(dataset_dir, 'images')
    ocr_dir = os.path.join(dataset_dir, 'box')
    entities_dir = os.path.join(dataset_dir, 'entities')

    # Collect all valid image files that have corresponding OCR and annotation files
    valid_files = []
    for file in os.listdir(entities_dir):
        if file.endswith('.txt'):
            file_name = os.path.splitext(file)[0]
            image_path = os.path.join(images_dir, file_name + '.jpg')
            ocr_path = os.path.join(ocr_dir, file)
            if os.path.exists(image_path) and os.path.exists(ocr_path):
                valid_files.append(file_name)

    # Shuffle the valid files and split into training and testing sets
    random.shuffle(valid_files)
    train_files = valid_files[:train_size]
    test_files = valid_files[train_size:]

    # Create directory structure for train and test datasets
    create_directory_structure(train_dir)
    create_directory_structure(test_dir)

    # Copy files to the train and test directories
    copy_files(train_files, dataset_dir, train_dir)
    copy_files(test_files, dataset_dir, test_dir)

    # Print statistics
    print(f"Total valid files: {len(valid_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")

if __name__ == "__main__":
    dataset_directory = "origDataset"
    train_directory = "CoRU_Train"
    test_directory = "CoRU_Test"
    train_size = 350  # Specify the number of images for the training set

    main(dataset_directory, train_directory, test_directory, train_size)
