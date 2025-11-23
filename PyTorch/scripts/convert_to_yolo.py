import os
from PIL import Image

# Paths
image_base = '../data/processed/images'   # train/val images
label_base = '../data/processed/labels'   # train/val labels

# Classes
classes = sorted(os.listdir(os.path.join(image_base, 'train')))
class_to_id = {cls_name: idx for idx, cls_name in enumerate(classes)}
print("Class mapping:", class_to_id)

# Make sure label folders exist
for split in ['train', 'val']:
    for cls_name in classes:
        os.makedirs(os.path.join(label_base, split, cls_name), exist_ok=True)

# Function to convert image to YOLO format
def convert_to_yolo(img_path, class_id, label_path):
    with Image.open(img_path) as img:
        w, h = img.size
    # Full image bounding box normalized
    x_center = 0.5
    y_center = 0.5
    width = 1.0
    height = 1.0
    # Write to label file
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Iterate splits and classes
for split in ['train', 'val']:
    for cls_name in classes:
        img_folder = os.path.join(image_base, split, cls_name)
        label_folder = os.path.join(label_base, split, cls_name)
        
        for img_file in os.listdir(img_folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(img_folder, img_file)
                label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.txt')
                convert_to_yolo(img_path, class_to_id[cls_name], label_path)

print("YOLO labels generated successfully!")
