import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms.functional as TF
import os

# Load Faster R-CNN model
def load_faster_rcnn_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)  # 3 classes
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, device

# Test detection on sample images
model_path = r"runs\detect\FasterRCNN_ResNet50_Optimized\weights\best.pt"
print(f"Loading model from: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

model, device = load_faster_rcnn_model(model_path)
print(f"✅ Model loaded successfully on {device}")

# Test on a sample image from your dataset
test_images = [
    r"datasets\palms_yolo\train\images\imgPalm220_JPG.rf.976cc46ef5bea5016dfdc85ffe9dd067.jpg",
    r"datasets\palms_yolo\valid\images\imgPalm307_JPG.rf.70e28e7d3d6872932d2d6ca076a804e5.jpg"
]

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        continue
    
    print(f"\n{'='*60}")
    print(f"Testing on: {os.path.basename(img_path)}")
    print(f"{'='*60}")
    
    image = Image.open(img_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Test at different resolutions
    resolutions = [512, 640, 800, 1024]
    
    for size in resolutions:
        img_resized = image.resize((size, size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img_resized).to(device)
        
        with torch.no_grad():
            predictions = model([img_tensor])
        
        pred = predictions[0]
        boxes = pred['boxes'].cpu()
        scores = pred['scores'].cpu()
        labels = pred['labels'].cpu()
        
        # Count detections at different confidence thresholds
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        print(f"\n  Size {size}x{size}:")
        for thresh in thresholds:
            count = len([s for s in scores if s > thresh])
            if count > 0:
                print(f"    Confidence > {thresh}: {count} detections")
                if thresh == 0.15:
                    # Show details for threshold 0.15
                    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                        if score > thresh:
                            cls_name = "Unhealthy" if label == 1 else "Healthy" if label == 2 else "Background"
                            print(f"      Detection {i+1}: {cls_name}, conf={score:.3f}, box={box.numpy()}")

print(f"\n{'='*60}")
print("Test complete!")
print(f"{'='*60}")
