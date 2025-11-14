import os
import cv2
import numpy as np
import shutil
import yaml
from pathlib import Path
import time
from datetime import datetime
import torch
import psutil
from ultralytics import YOLO
import albumentations as A
import warnings

warnings.filterwarnings('ignore')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PalmTrainingPipeline:
    """Complete training pipeline for palm health detection"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.dataset_path = "datasets/palms_yolo"
        self.original_train_img = f"{self.dataset_path}/train/images"
        self.original_train_lbl = f"{self.dataset_path}/train/labels"
        self.aug_train_img = f"{self.dataset_path}/train_aug/images"
        self.aug_train_lbl = f"{self.dataset_path}/train_aug/labels"
        self.dataset_yaml = f"{self.dataset_path}/dataset.yaml"
        
        self.model = None
        self.training_start_time = None
        
    def print_header(self, title):
        """Print a nice header"""
        print("\n" + "=" * 80)
        print(f"🎯 {title}")
        print("=" * 80)
    
    def check_system_specs(self):
        """Check and display system specifications"""
        self.print_header("SYSTEM SPECIFICATIONS")
        
        # RAM and CPU
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        print(f"💻 System Info:")
        print(f"   RAM: {ram_gb:.1f} GB")
        print(f"   CPU Cores: {cpu_count}")
        
        # GPU Check
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            device = 'cuda'
        else:
            print("   GPU: None (CPU training)")
            device = 'cpu'
        
        return device
    
    def check_original_data(self):
        """Check if original training data exists and count images"""
        self.print_header("ORIGINAL DATASET CHECK")
        
        if not os.path.exists(self.original_train_img):
            print(f"❌ Original training images not found: {self.original_train_img}")
            return False
            
        if not os.path.exists(self.original_train_lbl):
            print(f"❌ Original training labels not found: {self.original_train_lbl}")
            return False
        
        # Count images and labels
        images = [f for f in os.listdir(self.original_train_img) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        labels = [f for f in os.listdir(self.original_train_lbl) 
                 if f.endswith('.txt')]
        
        print(f"📊 Original Dataset:")
        print(f"   Images: {len(images)}")
        print(f"   Labels: {len(labels)}")
        
        if len(images) != len(labels):
            print(f"⚠️ Warning: Image-label count mismatch!")
        else:
            print(f"✅ Perfect image-label sync!")
        
        return len(images) > 0
    
    def fix_label_files(self):
        """Fix label files with tiny negative values and invalid ranges"""
        print(f"🔧 Fixing ALL label files for albumentations compatibility...")
        
        fixed_count = 0
        total_fixes = 0
        label_files = [f for f in os.listdir(self.original_train_lbl) if f.endswith('.txt')]
        
        for label_file in label_files:
            label_path = os.path.join(self.original_train_lbl, label_file)
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                fixed_lines = []
                file_needs_fix = False
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Store original for comparison
                        original = [x_center, y_center, width, height]
                        
                        # Aggressively fix any values outside [0, 1] range
                        # Round very small negative values to 0
                        if abs(x_center) < 1e-6 and x_center < 0:
                            x_center = 0.0
                        if abs(y_center) < 1e-6 and y_center < 0:
                            y_center = 0.0
                        if abs(width) < 1e-6 and width < 0:
                            width = 0.0
                        if abs(height) < 1e-6 and height < 0:
                            height = 0.0
                        
                        # Clamp to valid range [0, 1]
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        width = max(0.0, min(1.0, width))
                        height = max(0.0, min(1.0, height))
                        
                        # Check if we made any changes
                        if [x_center, y_center, width, height] != original:
                            file_needs_fix = True
                            total_fixes += 1
                        
                        # Always write with fixed precision to avoid scientific notation
                        fixed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                # Always write back to ensure consistent formatting
                with open(label_path, 'w') as f:
                    f.writelines(fixed_lines)
                
                if file_needs_fix:
                    fixed_count += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error fixing {label_file}: {e}")
        
        print(f"✅ Processed {len(label_files)} label files")
        print(f"✅ Fixed {fixed_count} files with {total_fixes} coordinate corrections")
        print(f"✅ All labels now compatible with albumentations")
        return fixed_count

    def create_augmented_data(self):
        """Create augmented training data from original dataset"""
        self.print_header("DATA AUGMENTATION PROCESS")
        
        # First, fix any problematic label files
        self.fix_label_files()
        
        # Create directories
        os.makedirs(self.aug_train_img, exist_ok=True)
        os.makedirs(self.aug_train_lbl, exist_ok=True)
        
        print(f"📂 Augmentation Setup:")
        print(f"   Source: {self.original_train_img}")
        print(f"   Target: {self.aug_train_img}")
        
        # Clear existing augmented data
        print(f"🧹 Clearing previous augmented data...")
        for f in Path(self.aug_train_img).glob("*"):
            f.unlink()
        for f in Path(self.aug_train_lbl).glob("*"):
            f.unlink()
        
        # Get original images
        original_images = [f for f in os.listdir(self.original_train_img) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📊 Processing {len(original_images)} original images...")
        
        # Copy original images and labels first
        copied_count = 0
        for img_file in original_images:
            try:
                # Copy image
                shutil.copy2(
                    os.path.join(self.original_train_img, img_file),
                    os.path.join(self.aug_train_img, img_file)
                )
                
                # Copy corresponding label
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_src = os.path.join(self.original_train_lbl, label_file)
                if os.path.exists(label_src):
                    shutil.copy2(label_src, os.path.join(self.aug_train_lbl, label_file))
                    copied_count += 1
                    
            except Exception as e:
                print(f"⚠️ Error copying {img_file}: {e}")
        
        print(f"✅ Copied {copied_count} original images")
        
        # Use simple OpenCV augmentation to avoid albumentations bbox issues
        print(f"🔄 Using OpenCV-based augmentation (more stable)...")
        
        # Create augmented versions
        print(f"🔄 Creating augmented versions...")
        augmented_count = 0
        
        for i, img_file in enumerate(original_images):
            try:
                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"   Processing: {i+1}/{len(original_images)} images...")
                
                # Read image
                img_path = os.path.join(self.original_train_img, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Read YOLO labels
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(self.original_train_lbl, label_file)
                
                bboxes = []
                class_labels = []
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # Fix tiny negative values that should be 0.0
                                x_center = max(0.0, min(1.0, x_center))
                                y_center = max(0.0, min(1.0, y_center))
                                width = max(0.0, min(1.0, width))
                                height = max(0.0, min(1.0, height))
                                
                                # Skip invalid bboxes
                                if width > 0 and height > 0:
                                    bboxes.append([x_center, y_center, width, height])
                                    class_labels.append(class_id)
                
                # Apply simple OpenCV-based augmentation (no bbox validation issues)
                try:
                    # Skip augmentation if no valid bboxes
                    if not bboxes:
                        continue
                    
                    # Apply augmentations directly with OpenCV
                    aug_image = image.copy()
                    aug_bboxes = [bbox[:] for bbox in bboxes]  # Copy bboxes
                    
                    # Random horizontal flip (50% chance)
                    if np.random.random() > 0.5:
                        aug_image = cv2.flip(aug_image, 1)
                        # Flip bboxes horizontally
                        for j in range(len(aug_bboxes)):
                            x_center = aug_bboxes[j][0]
                            aug_bboxes[j][0] = 1.0 - x_center  # Flip x coordinate
                    
                    # Random brightness (60% chance)
                    if np.random.random() > 0.4:
                        brightness = np.random.uniform(0.7, 1.3)
                        aug_image = cv2.convertScaleAbs(aug_image, alpha=brightness, beta=0)
                    
                    # Random contrast (60% chance)
                    if np.random.random() > 0.4:
                        contrast = np.random.uniform(0.8, 1.2)
                        aug_image = cv2.convertScaleAbs(aug_image, alpha=contrast, beta=0)
                    
                    # Random noise (30% chance)
                    if np.random.random() > 0.7:
                        noise = np.random.normal(0, 8, aug_image.shape).astype(np.int16)
                        aug_image = np.clip(aug_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        
                except Exception as aug_error:
                    # Skip this image if augmentation fails
                    print(f"   ⚠️ Skipping {img_file}: {str(aug_error)[:50]}...")
                    continue
                
                # Save augmented image
                name_base = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]
                aug_img_name = f"{name_base}_aug{ext}"
                aug_img_path = os.path.join(self.aug_train_img, aug_img_name)
                
                cv2.imwrite(aug_img_path, aug_image)
                
                # Save augmented labels
                aug_label_name = f"{name_base}_aug.txt"
                aug_label_path = os.path.join(self.aug_train_lbl, aug_label_name)
                
                with open(aug_label_path, 'w') as f:
                    for bbox, class_id in zip(aug_bboxes, class_labels):
                        x_center, y_center, width, height = bbox
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                augmented_count += 1
                
            except Exception as e:
                print(f"⚠️ Error augmenting {img_file}: {e}")
                continue
        
        total_images = copied_count + augmented_count
        increase_percent = ((total_images / copied_count) - 1) * 100 if copied_count > 0 else 0
        
        print(f"\n🎉 AUGMENTATION COMPLETE!")
        print(f"   Original images: {copied_count}")
        print(f"   Augmented images: {augmented_count}")
        print(f"   Total images: {total_images}")
        print(f"   Data increase: {increase_percent:.0f}%")
        
        return total_images
    
    def update_dataset_config(self):
        """Update dataset.yaml to use augmented training data"""
        self.print_header("DATASET CONFIGURATION UPDATE")
        
        if not os.path.exists(self.dataset_yaml):
            print(f"❌ Dataset config not found: {self.dataset_yaml}")
            return False
        
        # Read current config
        with open(self.dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update to use augmented data
        config['train'] = 'train_aug/images'
        
        # Write back
        with open(self.dataset_yaml, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Updated dataset.yaml:")
        print(f"   train: {config['train']}")
        print(f"   val: {config['val']}")
        print(f"   test: {config['test']}")
        
        return True
    
    def load_model(self, model_name='yolov8n.pt'):
        """Load YOLO model"""
        self.print_header("MODEL LOADING")
        
        print(f"📥 Loading {model_name}...")
        self.model = YOLO(model_name)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.model.parameters()) / 1e6
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: ~{total_params:.1f}M")
        print(f"   Architecture: {model_name.replace('.pt', '').upper()}")
        
        return True
    
    def estimate_training_time(self, epochs=150, batch_size=4, device='cpu'):
        """Estimate training time"""
        # Count augmented images
        if os.path.exists(self.aug_train_img):
            train_images = len([f for f in os.listdir(self.aug_train_img) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            train_images = 676  # Expected count
        
        # Time calculation
        steps_per_epoch = train_images // batch_size
        seconds_per_step = 1.5 if device == 'cuda' else 2.0  # GPU is faster
        
        epoch_time = steps_per_epoch * seconds_per_step
        total_hours = (epochs * epoch_time) / 3600
        
        print(f"⏱️ TRAINING TIME ESTIMATE:")
        print(f"   Training images: {train_images}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Image size: 640x640")
        print(f"   Device: {device.upper()}")
        print(f"   Time per epoch: ~{epoch_time/60:.1f} minutes")
        print(f"   Total estimated time: ~{total_hours:.1f} hours")
        print(f"   Target accuracy: 85-90% detection 🎯")
        
        return total_hours
    
    def train_model(self, device='cpu'):
        """Train the model with optimized parameters for 85-90% accuracy"""
        self.print_header("MODEL TRAINING")
        
        self.training_start_time = time.time()
        
        print(f"🚀 Starting training...")
        print(f"   Target: 85-90% detection accuracy")
        print(f"   Device: {device.upper()}")
        
        try:
            results = self.model.train(
                # Dataset
                data=self.dataset_yaml,
                
                # Training parameters - INCREASED for better accuracy
                epochs=150,      # More epochs for better convergence
                patience=30,     # More patience for finding optimal weights
                
                # Image and batch settings
                imgsz=640,       # Larger image size for better detection
                batch=8 if device == 'cuda' else 4,  # Larger batch if GPU available
                
                # Optimization - Fine-tuned for higher accuracy
                lr0=0.005,       # Lower initial learning rate for stability
                lrf=0.01,        # Final learning rate factor
                optimizer='AdamW',  # AdamW often performs better than SGD
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=5,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                cos_lr=True,     # Cosine learning rate scheduling
                
                # Enhanced augmentation for better generalization
                hsv_h=0.02,      # Slightly more hue variation
                hsv_s=0.8,       # More saturation variation
                hsv_v=0.5,       # More brightness variation
                degrees=15,      # More rotation for robustness
                translate=0.15,  # More translation
                scale=0.6,       # More scaling variation
                shear=2.0,       # Add shear transformation
                perspective=0.0001,  # Slight perspective change
                flipud=0.0,      # No vertical flip (palms don't appear upside down)
                fliplr=0.5,      # Keep horizontal flip
                mosaic=1.0,      # Full mosaic augmentation
                mixup=0.15,      # More mixup for better generalization
                copy_paste=0.15, # More copy-paste augmentation
                
                # Loss function weights for better detection
                box=7.5,         # Box loss weight
                cls=0.5,         # Class loss weight
                dfl=1.5,         # Distribution focal loss weight
                
                # Model regularization
                dropout=0.0,     # No dropout for small model
                
                # Training settings
                name='YOLO_Detection',
                project='runs/detect',
                save=True,
                save_period=20,  # Save every 20 epochs
                plots=True,
                
                # Hardware settings
                device=device,
                workers=4 if device == 'cuda' else 2,
                
                # Memory optimization
                val=True,
                cache=False,     # No caching to save memory
                verbose=True,
                amp=True if device == 'cuda' else False,  # Automatic Mixed Precision for GPU

                # Reproducibility
                seed=42,
                deterministic=True,
            )
            
            training_time = (time.time() - self.training_start_time) / 3600
            
            print(f"\n🎉 TRAINING COMPLETED!")
            print(f"   Training time: {training_time:.2f} hours")
            print(f"   Results saved to: runs/detect/{results.save_dir}")
            
            return results
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            print(f"\n💡 Troubleshooting tips:")
            print(f"   1. Check available disk space (need ~3GB)")
            print(f"   2. Try reducing batch size to 2 if memory error")
            print(f"   3. Close other applications to free RAM")
            print(f"   4. Consider using a GPU for faster training")
            return None
    

    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("🎯 PALM HEALTH DETECTION - COMPLETE TRAINING PIPELINE")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Goal: Achieve 85-90% detection accuracy with optimized training")
        print(f"Key improvements:")
        print(f"   - Larger image size (640x640 vs 416x416)")
        print(f"   - More epochs (150 vs 80) with AdamW optimizer")
        print(f"   - Enhanced augmentation and regularization")
        print(f"   - Tuned loss weights for better detection")
        
        # Step 1: Check system
        device = self.check_system_specs()
        
        # Step 2: Check original data
        if not self.check_original_data():
            print("❌ Pipeline failed: Original data not found")
            return False
        
        # Step 3: Create augmented data
        total_images = self.create_augmented_data()
        if total_images == 0:
            print("❌ Pipeline failed: Data augmentation failed")
            return False
        
        # Step 4: Update dataset config
        if not self.update_dataset_config():
            print("❌ Pipeline failed: Dataset config update failed")
            return False
        
        # Step 5: Load model
        if not self.load_model('yolov8n.pt'):
            print("❌ Pipeline failed: Model loading failed")
            return False
        
        # Step 6: Training time estimate
        estimated_time = self.estimate_training_time(device=device)
        
        # Ask for confirmation
        print(f"\n🤔 Ready to start training (~{estimated_time:.1f} hours)")
        response = input("Continue with training? (y/n): ").lower().strip()
        
        if response != 'y' and response != 'yes':
            print("❌ Training cancelled by user")
            return False
        
        # Step 7: Train model
        results = self.train_model(device=device)
        if results is None:
            print("❌ Pipeline failed: Training failed")
            return False
        
        # Training completed successfully
        model_path = f"runs/detect/{results.save_dir}/weights/best.pt"
        training_time = (time.time() - self.training_start_time) / 3600
        
        print(f"\n🎉 TRAINING PIPELINE COMPLETED!")
        print(f"   Training time: {training_time:.2f} hours")
        print(f"   Model saved to: {model_path}")
        print(f"   Target accuracy: 85-90% detection")
        print(f"\n📋 Next Steps:")
        print(f"   1. Run: python evaluation_metrics.py")
        print(f"   2. Check detection accuracy in evaluation results")
        print(f"   3. Update stm.py with new model path if accuracy improved")
        print(f"   4. Test inference on new palm images")
        
        print(f"\n✅ COMPLETE PIPELINE FINISHED!")
        return True

def main():
    """Main function to run the training pipeline"""
    pipeline = PalmTrainingPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print(f"\n🎉 All done! Your palm health detection model is ready!")
    else:
        print(f"\n❌ Pipeline encountered issues. Check the output above.")

if __name__ == "__main__":
    main()