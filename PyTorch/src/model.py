from ultralytics import YOLO

def get_model(pretrained=True, num_classes=9):
    """
    Load YOLOv12-seg model.
    pretrained: whether to use pretrained weights
    num_classes: number of skin disease classes
    """
    # Use YOLOv8n-seg as base (smallest, fast for testing)
    model = YOLO('yolov8n-seg.pt') if pretrained else YOLO('yolov8n-seg.yaml')
    
    # Update number of classes
    model.model[-1].nc = num_classes
    return model
