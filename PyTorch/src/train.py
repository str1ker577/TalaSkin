from dataset import SkinYoloDataset
from model import get_model

# Paths
data_dir = '../data/processed/'

# Initialize dataset
dataset = SkinYoloDataset(data_dir)
yolo_paths = dataset.get_yolo_paths()

# Initialize model
num_classes = 9  # adjust based on dataset
model = get_model(pretrained=True, num_classes=num_classes)

# Training
model.train(
    data=yolo_paths,
    epochs=10,
    imgsz=640,
    batch=8,
    name='tala_skin_seg'
)

print("Training complete. Model weights saved in 'runs/train/tala_skin_seg'")
