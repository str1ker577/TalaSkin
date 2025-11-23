import os
import shutil
import random

# Paths
raw_train_dir = '../data/raw/SkinDisease/Train'  # original Kaggle train folder
processed_image_train = '../data/processed/images/train'
processed_image_val = '../data/processed/images/val'

# Create directories if they don't exist
os.makedirs(processed_image_train, exist_ok=True)
os.makedirs(processed_image_val, exist_ok=True)

# Split ratio
split_ratio = 0.8

# Iterate through each class folder
for class_name in os.listdir(raw_train_dir):
    class_path = os.path.join(raw_train_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    images = os.listdir(class_path)
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    # Create class folders in processed folder
    os.makedirs(os.path.join(processed_image_train, class_name), exist_ok=True)
    os.makedirs(os.path.join(processed_image_val, class_name), exist_ok=True)
    
    # Copy images to train folder
    for img in train_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(processed_image_train, class_name, img))
    
    # Copy images to val folder
    for img in val_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(processed_image_val, class_name, img))

print("Dataset split completed!")
