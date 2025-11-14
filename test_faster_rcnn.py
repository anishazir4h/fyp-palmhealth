import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms.functional as TF
import os

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "runs/detect/FasterRCNN_ResNet50_Optimized/weights/best.pt"

print("Loading model...")
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)  # background + 2 classes

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully\n")

# Try multiple test images
test_images = [
    "datasets/test/imgPalm100_JPG.rf.3f6481517add1b8c09dfe6f88b036727.jpg",
    "datasets/train/imgPalm1_JPG.rf.031ac2e2dd6bef5e1fb8e3f93cbd7dcb.jpg",
    "datasets/train_aug/imgPalm1_JPG.rf.031ac2e2dd6bef5e1fb8e3f93cbd7dcb.jpg"
]

for test_image_path in test_images:
    if not os.path.exists(test_image_path):
        print(f"⏭️  Skipping {test_image_path} (not found)")
        continue
    
    print(f"\n{'='*80}")
    print(f"Testing: {os.path.basename(test_image_path)}")
    print(f"{'='*80}")
    
    image = Image.open(test_image_path).convert("RGB")
    print(f"📷 Original image size: {image.size}")
    
    # Resize to 512x512 (same as training)
    resized_image = image.resize((512, 512), Image.BILINEAR)
    img_tensor = TF.to_tensor(resized_image).to(device)
    
    print(f"🔍 Running inference...")
    
    # Run inference
    with torch.no_grad():
        predictions = model([img_tensor])
    
    # Check predictions
    pred = predictions[0]
    boxes = pred['boxes'].cpu()
    scores = pred['scores'].cpu()
    labels = pred['labels'].cpu()
    
    print(f"\n📊 Results:")
    print(f"   Total predictions: {len(scores)}")
    if len(scores) > 0:
        print(f"   Score range: {scores.min():.4f} - {scores.max():.4f}")
        print(f"   Unique labels: {labels.unique().tolist()}")
        print(f"\n   Top predictions:")
        for i in range(min(5, len(scores))):
            print(f"   {i+1}. Score: {scores[i]:.4f}, Label: {labels[i].item()}")
        
        # Check different thresholds
        for threshold in [0.05, 0.10, 0.25, 0.50]:
            count = (scores >= threshold).sum().item()
            print(f"\n   Threshold {threshold:.2f}: {count} detections")
    else:
        print("   ❌ ZERO predictions!")

print(f"\n\n{'='*80}")
print("DIAGNOSIS:")
print("If ALL images show 0 predictions, the model is broken/incompatible.")
print("If SOME images work, the model was trained on different image types.")
print(f"{'='*80}")
