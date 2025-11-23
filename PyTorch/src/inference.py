from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# Load trained model
model_path = '../runs/train/tala_skin_seg/weights/best.pt'
model = YOLO(model_path)

# Test image
image_path = '../data/processed/images/val/Acne/123.jpg'

# Run prediction
results = model.predict(image_path, imgsz=640)

# Display results
results[0].show()  # opens an image window with masks overlaid
results[0].save(Path('../runs/inference'))
