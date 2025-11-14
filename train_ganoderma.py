"""
Faster R-CNN Training for Ganoderma Disease Detection
Using Ganoderma COCO Dataset for Plantation Scene Context Validation

Dataset: Ganoderma Detection Dataset for Oil Palm Crop Disease Classification
Purpose: Train a specialized model for validating palms in aerial/satellite imagery
         and detecting Ganoderma disease patterns
"""

import os
import json
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import psutil
import gc
from tqdm import tqdm

class GanodermaCocoDataset(Dataset):
    """Dataset for Ganoderma detection using COCO format annotations"""
    
    def __init__(self, coco_json_path, img_dir, transforms=None, img_size=640):
        """
        Args:
            coco_json_path: Path to COCO format JSON file
            img_dir: Directory containing images
            transforms: Optional transforms
            img_size: Target image size (default: 640, matching Ganoderma dataset)
        """
        # Load COCO annotations
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_size = img_size
        
        # Build image id to annotations mapping
        self.img_id_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
        
        # Get images that have annotations (only diseased palms are annotated)
        self.images = [img for img in self.coco_data['images'] 
                      if img['id'] in self.img_id_to_anns]
        
        # Class mapping: 0 = background, 1 = Ganoderma
        self.categories = self.coco_data['categories']
        print(f"Dataset: {len(self.images)} images with Ganoderma annotations")
        print(f"Total annotations: {len(self.coco_data['annotations'])}")
        print(f"Categories: {self.categories}")
        print(f"Image size: {img_size}×{img_size}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_id = img_info['id']
        img_name = img_info['file_name']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"Warning: Could not load {img_path}, using blank image")
            image = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        
        # Store original dimensions for bbox scaling
        orig_width, orig_height = image.size
        
        # Resize image (Ganoderma dataset uses 640x640)
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Calculate scaling factors for bounding boxes
        scale_x = self.img_size / orig_width
        scale_y = self.img_size / orig_height
        
        # Get annotations for this image
        annotations = self.img_id_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        
        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Convert to [xmin, ymin, xmax, ymax] and scale
            xmin = x * scale_x
            ymin = y * scale_y
            xmax = (x + w) * scale_x
            ymax = (y + h) * scale_y
            
            # Ensure valid box
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                # category_id in COCO is 0 or 1, both are "Ganoderma"
                # We use 1 for Ganoderma (0 is background)
                labels.append(1)  # Ganoderma class
        
        # Handle images with no valid boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        image_id = torch.tensor([img_id])
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        # Convert image to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target


def collate_fn(batch):
    """Custom collate function for dataloader"""
    return tuple(zip(*batch))


class GanodermaFasterRCNNTrainer:
    """Trainer for Ganoderma-specific Faster R-CNN"""
    
    def __init__(self, num_classes=2):
        """
        Args:
            num_classes: Number of classes including background (default: 2)
                        0 = background, 1 = Ganoderma
        """
        self.num_classes = num_classes
        self.model = None
        self.device = None
        
    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "=" * 80)
        print(f"🔬 {title}")
        print("=" * 80)
        
    def check_system_specs(self):
        """Check and display system specifications"""
        self.print_header("SYSTEM SPECIFICATIONS")
        
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        print(f"💻 System Info:")
        print(f"   RAM: {ram_gb:.1f} GB")
        print(f"   CPU Cores: {cpu_count}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            device = 'cuda'
        else:
            print("   GPU: None (CPU training)")
            device = 'cpu'
        
        self.device = torch.device(device)
        return device
    
    def create_model(self):
        """Create Faster R-CNN model for Ganoderma detection"""
        self.print_header("MODEL CREATION")
        
        print(f"📥 Creating Faster R-CNN with ResNet-50 backbone")
        print(f"   Number of classes: {self.num_classes} (0=Background, 1=Ganoderma)")
        print(f"   Purpose: Plantation scene validation + Disease detection")
        
        # Load pretrained Faster R-CNN with ResNet-50
        # Optimized for 640×640 images (matching Ganoderma dataset)
        model = fasterrcnn_resnet50_fpn(
            pretrained=False, 
            pretrained_backbone=True,
            min_size=640,
            max_size=640,
            box_detections_per_img=100,
            rpn_pre_nms_top_n_train=1000,
            rpn_post_nms_top_n_train=1000
        )
        
        # Replace the classifier head for Ganoderma detection
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        self.model = model
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        
        print(f"✅ Model created successfully!")
        print(f"   Total parameters: ~{total_params:.1f}M")
        print(f"   Trainable parameters: ~{trainable_params:.1f}M")
        print(f"   Architecture: Faster R-CNN + ResNet-50-FPN")
        
        return model
    
    def train(self, train_json, train_img_dir, valid_json, valid_img_dir, 
              epochs=30, batch_size=2, lr=0.005, save_dir="runs/detect/FasterRCNN_Ganoderma"):
        """
        Train Faster R-CNN for Ganoderma disease detection
        
        Args:
            train_json: Path to training COCO JSON file
            train_img_dir: Directory with training images
            valid_json: Path to validation COCO JSON file
            valid_img_dir: Directory with validation images
            epochs: Number of training epochs (default: 30)
            batch_size: Batch size (default: 2)
            lr: Learning rate
            save_dir: Directory to save model weights
        """
        self.print_header("TRAINING GANODERMA FASTER R-CNN")
        
        # Check device
        if self.device is None:
            self.check_system_specs()
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Move model to device
        self.model.to(self.device)
        
        # Create datasets
        print(f"\n📊 Loading Ganoderma COCO datasets...")
        train_dataset = GanodermaCocoDataset(train_json, train_img_dir, img_size=640)
        valid_dataset = GanodermaCocoDataset(valid_json, valid_img_dir, img_size=640)
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(valid_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        best_loss = float('inf')
        checkpoint_path = Path("runs/detect/FasterRCNN_Ganoderma/weights/last.pt")
        
        # If last.pt doesn't exist, try best.pt
        if not checkpoint_path.exists():
            checkpoint_path = Path("runs/detect/FasterRCNN_Ganoderma/weights/best.pt")
        
        if checkpoint_path.exists():
            print(f"\n📂 Found existing checkpoint: {checkpoint_path}")
            resume = input("   Resume from last checkpoint? (y/n): ").strip().lower()
            
            if resume == 'y':
                print(f"   Loading checkpoint...")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # New checkpoint format with full state
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    best_loss = checkpoint.get('best_loss', float('inf'))
                    print(f"   ✅ Resumed from epoch {start_epoch}")
                    print(f"   Best loss so far: {best_loss:.4f}")
                else:
                    # Old format - just model weights
                    self.model.load_state_dict(checkpoint)
                    print(f"   ✅ Loaded model weights (optimizer state reset)")
                    print(f"   ⚠️ Note: Starting from epoch 0 (no training state found)")
        
        print(f"\n�🚀 Starting training...")
        print(f"   Epochs: {start_epoch} → {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {lr}")
        print(f"   Device: {self.device}")
        print(f"   Purpose: Scene validation + Ganoderma disease detection")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            # Progress bar for training
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for images, targets in pbar:
                try:
                    # Filter out samples with no boxes
                    valid_samples = [(img, tgt) for img, tgt in zip(images, targets) 
                                   if len(tgt['boxes']) > 0]
                    
                    if len(valid_samples) == 0:
                        continue
                    
                    images, targets = zip(*valid_samples)
                    
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Forward pass
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Backward pass
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': f'{losses.item():.4f}'})
                    
                    # Clear memory periodically
                    if num_batches % 50 == 0:
                        gc.collect()
                        
                except RuntimeError as e:
                    print(f"\n⚠️ Warning: Batch skipped due to error: {e}")
                    gc.collect()
                    continue
            
            # Update learning rate
            lr_scheduler.step()
            
            # Calculate average loss
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # Validation
            if (epoch + 1) % 1 == 0:
                self.model.train()  # Keep in train mode to get loss
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for images, targets in valid_loader:
                        # Filter out samples with no boxes
                        valid_samples = [(img, tgt) for img, tgt in zip(images, targets) 
                                       if len(tgt['boxes']) > 0]
                        
                        if len(valid_samples) == 0:
                            continue
                        
                        images, targets = zip(*valid_samples)
                        
                        images = list(image.to(self.device) for image in images)
                        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                        
                        loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        val_loss += losses.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Save best model
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    save_path = Path(save_dir) / "weights"
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save full checkpoint with training state
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'best_loss': best_loss,
                        'train_loss': avg_loss,
                        'val_loss': avg_val_loss
                    }
                    torch.save(checkpoint, save_path / "best.pt")
                    print(f"      💾 Saved best model (Val Loss: {best_loss:.4f})")
            else:
                print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
        
        # Save final model with full state
        save_path = Path(save_dir) / "weights"
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epochs - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_loss': best_loss,
            'final_epoch': True
        }
        torch.save(checkpoint, save_path / "last.pt")
        
        training_time = (time.time() - start_time) / 3600
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"   Training time: {training_time:.2f} hours")
        print(f"   Best validation loss: {best_loss:.4f}")
        print(f"   Models saved to: {save_dir}/weights/")
        
        return self.model


def main():
    """Main training function"""
    print("🔬 FASTER R-CNN TRAINING FOR GANODERMA DISEASE DETECTION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: Ganoderma Detection Dataset for Oil Palm Crop Disease Classification")
    print(f"Purpose: Two-stage detection system")
    print(f"   Stage 1: YOLO (individual palm detection + health classification)")
    print(f"   Stage 2: This model (scene validation + Ganoderma disease detection)")
    
    # Dataset paths
    base_dir = r"datasets\ganoderma\Ganoderma Detection Dataset for Oil Palm Crop Disease Classification"
    train_json = os.path.join(base_dir, "train", "_annotations.coco.json")
    train_img_dir = os.path.join(base_dir, "train")
    valid_json = os.path.join(base_dir, "valid", "_annotations.coco.json")
    valid_img_dir = os.path.join(base_dir, "valid")
    
    # Check if files exist
    if not os.path.exists(train_json):
        print(f"❌ Error: Training JSON not found at {train_json}")
        return
    
    if not os.path.exists(valid_json):
        print(f"❌ Error: Validation JSON not found at {valid_json}")
        return
    
    # Create trainer
    trainer = GanodermaFasterRCNNTrainer(num_classes=2)  # 0=background, 1=Ganoderma
    
    # Check system
    trainer.check_system_specs()
    
    print("\n🎯 TRAINING OBJECTIVES:")
    print("   1. Validate YOLO detections in plantation scenes")
    print("   2. Detect Ganoderma disease (override health classification)")
    print("   3. Filter false positives from aerial/satellite imagery")
    print("   4. Understand tree patterns in plantation context")
    print("")
    print("🔬 MODEL CAPABILITIES AFTER TRAINING:")
    print("   ✓ Validates if detection is a real palm in plantation scene")
    print("   ✓ Detects Ganoderma disease presence")
    print("   ✓ Better at handling aerial/satellite imagery")
    print("   ✓ Complements YOLO's individual palm expertise")
    print("")
    print("📊 DATASET INFO:")
    print("   - 987 training images with Ganoderma annotations")
    print("   - 1,215 disease bounding boxes")
    print("   - 640×640 pixel images (aerial/satellite views)")
    print("   - Only diseased palms are annotated")
    print("")
    print("⚡ TRAINING STRATEGY:")
    print("   - 30 epochs (sufficient for disease pattern learning)")
    print("   - Batch size: 2 (stable training)")
    print("   - Learning rate: 0.005 with decay")
    print("   - Expected time: ~10-15 hours")
    
    trainer.train(
        train_json=train_json,
        train_img_dir=train_img_dir,
        valid_json=valid_json,
        valid_img_dir=valid_img_dir,
        epochs=30,
        batch_size=2,
        lr=0.005,
        save_dir="runs/detect/FasterRCNN_Ganoderma"
    )
    
    print(f"\n✅ Training pipeline completed!")
    print(f"\n📋 Next steps:")
    print(f"   1. Integrate this model into stm.py for validation")
    print(f"   2. Update validate_with_faster_rcnn_2batch() to use this model")
    print(f"   3. Implement Ganoderma disease override logic")
    print(f"   4. Test on satellite/aerial images")
    print(f"\n🔄 Model Integration:")
    print(f"   Model path: runs/detect/FasterRCNN_Ganoderma/weights/best.pt")
    print(f"   Use for: Scene validation + Ganoderma detection")
    print(f"   Override rule: If Ganoderma detected → Change to UNHEALTHY (red box)")


if __name__ == "__main__":
    main()
