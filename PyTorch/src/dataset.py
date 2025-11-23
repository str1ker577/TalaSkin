import os
from pathlib import Path
from ultralytics import YOLO

# Placeholder for YOLOv12 training; dataset should be in YOLO format
class SkinYoloDataset:
    def __init__(self, data_dir):
        """
        data_dir: folder containing 'images/train', 'images/val', 'labels/train', 'labels/val'
        """
        self.data_dir = Path(data_dir)
        self.train_images = self.data_dir / 'images/train'
        self.val_images = self.data_dir / 'images/val'
        self.train_labels = self.data_dir / 'labels/train'
        self.val_labels = self.data_dir / 'labels/val'

    def get_yolo_paths(self):
        return {
            'train': str(self.train_images),
            'val': str(self.val_images),
        }
