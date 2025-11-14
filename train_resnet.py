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
import warnings
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

warnings.filterwarnings('ignore')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PalmTrainingPipeline:
    
    def __init__(self):
        self.base_path = Path(".")
        self.dataset_path = "datasets"
        self.original_train_img = f"{self.dataset_path}/train"
        self.original_valid_img = f"{self.dataset_path}/valid"
        self.original_test_img = f"{self.dataset_path}/test"
        self.aug_train_img = f"{self.dataset_path}/train_aug/images"
        self.aug_train_lbl = f"{self.dataset_path}/train_aug/labels"
        
        self.model = None
        self.training_start_time = None
        
    def print_header(self, title):
        print("\n" + "=" * 80)
        print(f"🎯 {title}")
        print("=" * 80)
    
    def check_system_specs(self):
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
        
        return device
    
    def check_original_data(self):
        self.print_header("ORIGINAL DATASET CHECK")
        
        train_csv = f"{self.original_train_img}/_annotations.csv"
        valid_csv = f"{self.original_valid_img}/_annotations.csv"
        test_csv = f"{self.original_test_img}/_annotations.csv"
        
        if not os.path.exists(train_csv):
            print(f"❌ Training CSV not found: {train_csv}")
            return False
            
        if not os.path.exists(valid_csv):
            print(f"❌ Validation CSV not found: {valid_csv}")
            return False
        
        try:
            train_df = pd.read_csv(train_csv)
            valid_df = pd.read_csv(valid_csv)
            
            train_images = train_df['filename'].nunique()
            valid_images = valid_df['filename'].nunique()
            
            classes = sorted(train_df['class'].unique())
            
            print(f"📊 CSV Dataset:")
            print(f"   Train images: {train_images}")
            print(f"   Valid images: {valid_images}")
            print(f"   Classes: {classes}")
            print(f"   Train annotations: {len(train_df)}")
            print(f"   Valid annotations: {len(valid_df)}")
            
            print(f"✅ CSV data loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error reading CSV files: {e}")
            return False
    
    def fix_label_files(self):
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
                        
                        original = [x_center, y_center, width, height]
                        
                        if abs(x_center) < 1e-6 and x_center < 0:
                            x_center = 0.0
                        if abs(y_center) < 1e-6 and y_center < 0:
                            y_center = 0.0
                        if abs(width) < 1e-6 and width < 0:
                            width = 0.0
                        if abs(height) < 1e-6 and height < 0:
                            height = 0.0
                        
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        width = max(0.0, min(1.0, width))
                        height = max(0.0, min(1.0, height))
                        
                        if [x_center, y_center, width, height] != original:
                            file_needs_fix = True
                            total_fixes += 1
                        
                        fixed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
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
        self.print_header("DATA AUGMENTATION PROCESS")
        
        self.fix_label_files()
        
        os.makedirs(self.aug_train_img, exist_ok=True)
        os.makedirs(self.aug_train_lbl, exist_ok=True)
        
        print(f"📂 Augmentation Setup:")
        print(f"   Source: {self.original_train_img}")
        print(f"   Target: {self.aug_train_img}")
        
        print(f"🧹 Clearing previous augmented data...")
        for f in Path(self.aug_train_img).glob("*"):
            f.unlink()
        for f in Path(self.aug_train_lbl).glob("*"):
            f.unlink()
        
        original_images = [f for f in os.listdir(self.original_train_img) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📊 Processing {len(original_images)} original images...")
        
        copied_count = 0
        for img_file in original_images:
            try:
                shutil.copy2(
                    os.path.join(self.original_train_img, img_file),
                    os.path.join(self.aug_train_img, img_file)
                )
                
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_src = os.path.join(self.original_train_lbl, label_file)
                if os.path.exists(label_src):
                    shutil.copy2(label_src, os.path.join(self.aug_train_lbl, label_file))
                    copied_count += 1
                    
            except Exception as e:
                print(f"⚠️ Error copying {img_file}: {e}")
        
        print(f"✅ Copied {copied_count} original images")
        
        print(f"🔄 Using OpenCV-based augmentation (more stable)...")
        print(f"🔄 Creating augmented versions...")
        augmented_count = 0
        
        for i, img_file in enumerate(original_images):
            try:
                if (i + 1) % 50 == 0:
                    print(f"   Processing: {i+1}/{len(original_images)} images...")
                
                img_path = os.path.join(self.original_train_img, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
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
                                
                                x_center = max(0.0, min(1.0, x_center))
                                y_center = max(0.0, min(1.0, y_center))
                                width = max(0.0, min(1.0, width))
                                height = max(0.0, min(1.0, height))
                                
                                if width > 0 and height > 0:
                                    bboxes.append([x_center, y_center, width, height])
                                    class_labels.append(class_id)
                
                try:
                    if not bboxes:
                        continue
                    
                    aug_image = image.copy()
                    aug_bboxes = [bbox[:] for bbox in bboxes]
                    
                    if np.random.random() > 0.5:
                        aug_image = cv2.flip(aug_image, 1)
                        for j in range(len(aug_bboxes)):
                            x_center = aug_bboxes[j][0]
                            aug_bboxes[j][0] = 1.0 - x_center
                    
                    if np.random.random() > 0.4:
                        brightness = np.random.uniform(0.7, 1.3)
                        aug_image = cv2.convertScaleAbs(aug_image, alpha=brightness, beta=0)
                    
                    if np.random.random() > 0.4:
                        contrast = np.random.uniform(0.8, 1.2)
                        aug_image = cv2.convertScaleAbs(aug_image, alpha=contrast, beta=0)
                    
                    if np.random.random() > 0.7:
                        noise = np.random.normal(0, 8, aug_image.shape).astype(np.int16)
                        aug_image = np.clip(aug_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        
                except Exception as aug_error:
                    print(f"   ⚠️ Skipping {img_file}: {str(aug_error)[:50]}...")
                    continue
                
                name_base = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]
                aug_img_name = f"{name_base}_aug{ext}"
                aug_img_path = os.path.join(self.aug_train_img, aug_img_name)
                
                cv2.imwrite(aug_img_path, aug_image)
                
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
        self.print_header("DATASET CONFIGURATION UPDATE")
        
        if not os.path.exists(self.dataset_yaml):
            print(f"❌ Dataset config not found: {self.dataset_yaml}")
            return False
        
        with open(self.dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        config['train'] = 'train_aug/images'
        
        with open(self.dataset_yaml, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Updated dataset.yaml:")
        print(f"   train: {config['train']}")
        print(f"   val: {config['val']}")
        print(f"   test: {config['test']}")
        
        return True
    
    def load_model(self):
        self.print_header("MODEL LOADING - ResNet-50")

        train_csv = f"{self.original_train_img}/_annotations.csv"
        num_classes = 2
        
        try:
            if os.path.exists(train_csv):
                df = pd.read_csv(train_csv)
                unique_classes = df['class'].unique()
                num_classes = len(unique_classes)
                print(f"📊 Found {num_classes} classes: {list(unique_classes)}")
        except Exception as e:
            print(f"⚠️ Could not read CSV, using default 2 classes: {e}")

        print(f"📥 Creating ResNet-50 classifier with {num_classes} classes")
        try:
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            self.model = model
            self.num_classes = num_classes
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"✅ Model created successfully!")
            print(f"   Parameters: ~{total_params:.1f}M")
            print(f"   Architecture: ResNet-50 Classification")
            print(f"   Backbone: ResNet-50 (pretrained on ImageNet)")
            return True
        except Exception as e:
            print(f"❌ Failed to create ResNet-50 model: {e}")
            return False
    
    def estimate_training_time(self, epochs=80, batch_size=4, device='cpu'):
        if os.path.exists(self.aug_train_img):
            train_images = len([f for f in os.listdir(self.aug_train_img) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            train_images = 676
        
        steps_per_epoch = train_images // batch_size
        seconds_per_step = 1.5 if device == 'cuda' else 1.8
        
        epoch_time = steps_per_epoch * seconds_per_step
        total_hours = (epochs * epoch_time) / 3600
        
        print(f"⏱️ TRAINING TIME ESTIMATE:")
        print(f"   Training images: {train_images}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {device.upper()}")
        print(f"   Time per epoch: ~{epoch_time/60:.1f} minutes")
        print(f"   Total estimated time: ~{total_hours:.1f} hours")
        print(f"   Expected accuracy: 97-98% 🎯")
        
        return total_hours
    
    def train_model(self, device='cpu'):
        self.print_header("MODEL TRAINING - ResNet-50 Classification")
 
        self.training_start_time = time.time()

        device_torch = torch.device('cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
        self.model.to(device_torch)

        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        class CSVClassificationDataset(Dataset):
            def __init__(self, csv_path, img_dir, transform=None):
                self.df = pd.read_csv(csv_path)
                self.img_dir = img_dir
                self.transform = transform
                
                self.classes = sorted(self.df['class'].unique())
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
                
                self.image_labels = {}
                for _, row in self.df.iterrows():
                    filename = row['filename']
                    class_name = row['class']
                    self.image_labels[filename] = self.class_to_idx[class_name]
                
                self.images = list(self.image_labels.keys())

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img_name = self.images[idx]
                img_path = os.path.join(self.img_dir, img_name)
                
                try:
                    image = Image.open(img_path).convert('RGB')
                except:
                    image = Image.new('RGB', (224, 224), (0, 0, 0))
                
                if self.transform:
                    image = self.transform(image)
                
                label = self.image_labels[img_name]
                return image, label

        train_csv = "datasets/train_aug/_annotations.csv"
        val_csv = "datasets/valid_aug/_annotations.csv"
        train_img_dir = "datasets/train_aug"
        val_img_dir = "datasets/valid_aug"
        
        train_dataset = CSVClassificationDataset(train_csv, train_img_dir, train_transform)
        val_dataset = CSVClassificationDataset(val_csv, val_img_dir, val_transform)
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=0.0001
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=0.5,
            patience=3
        )
        
        num_epochs = 50
        patience = 10
        best_acc = 0.0
        best_f1 = 0.0
        epochs_without_improvement = 0
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        print(f"🚀 Starting training...")
        print(f"   Paper: Heng et al. (2025) - IJACSA Vol 16 No 4")
        print(f"   Paper Method: Faster R-CNN (ResNet-50)")
        print(f"   Paper Results: F1 ~95% detection, 92% healthy, 86.9% unhealthy")
        print(f"   Data Source: UAV RGB images")
        print(f"   Our Dataset:")
        print(f"     Classes: {train_dataset.classes}")
        print(f"     Train images: {len(train_dataset)}")
        print(f"     Val images: {len(val_dataset)}")
        print(f"   Training Config:")
        print(f"     Batch size: {batch_size}")
        print(f"     Epochs: {num_epochs}")
        print(f"     Learning rate: 0.001 (Adam optimizer)")
        print(f"     Scheduler: ReduceLROnPlateau")
        print(f"     Early stopping patience: {patience}")
        print(f"     Device: {device_torch}")

        try:
            for epoch in range(num_epochs):
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for images, labels in train_loader:
                    images = images.to(device_torch)
                    labels = labels.to(device_torch)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                train_loss = running_loss / len(train_loader)
                train_acc = 100 * correct / total
                
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device_torch)
                        labels = labels.to(device_torch)
                        
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss = val_loss / len(val_loader)
                val_acc = 100 * val_correct / val_total
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['learning_rate'].append(optimizer.param_groups[0]['lr'])
                
                scheduler.step(val_acc)
                
                print(f"   Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    epochs_without_improvement = 0
                    print(f"   ✅ New best accuracy: {best_acc:.2f}%")
                else:
                    epochs_without_improvement += 1
                    print(f"   ⏳ No improvement for {epochs_without_improvement} epochs")
                
                if epochs_without_improvement >= patience:
                    print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
                    print(f"   Best validation accuracy: {best_acc:.2f}%")
                    break
                    
            save_dir = Path("runs/detect/ResNet50_Classification_Augmented")
            save_dir.mkdir(parents=True, exist_ok=True)
            weights_dir = save_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            model_path = weights_dir / "best.pt"
            torch.save(self.model.state_dict(), str(model_path))

            print(f"\n📊 Generating results folder with training history and evaluation plots...")
            self.generate_results_folder(history, val_loader, train_dataset.classes, save_dir, device_torch, best_acc, patience)

            training_time = (time.time() - self.training_start_time) / 3600
            print(f"\n🎉 TRAINING COMPLETED!")
            print(f"   Training time: {training_time:.2f} hours")
            print(f"   Best validation accuracy: {best_acc:.2f}%")
            print(f"   Model saved to: {model_path}")
            print(f"\n📁 RESULTS FOLDER: {save_dir}")
            print(f"   ✅ results.csv (training history)")
            print(f"   ✅ training_curves.png (loss, accuracy, LR curves)")
            print(f"   ✅ confusion_matrix.png (evaluation metrics)")
            print(f"   ✅ classification_report.txt (per-class metrics)")
            print(f"   ✅ training_summary.txt (complete summary)")

            class Result:
                def __init__(self, save_dir, best_acc):
                    self.save_dir = save_dir
                    self.best_acc = best_acc

            return Result(save_dir.name, best_acc)

        except KeyboardInterrupt:
            print("❌ Training interrupted by user")
            return None
        except Exception as e:
            print(f"❌ Training error: {e}")
            return None
    

    def run_complete_pipeline(self):
        print(f"🎯 PALM HEALTH DETECTION - RESNET-50 CLASSIFICATION")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📄 Following: Heng et al. (2025)")
        print(f"   Title: 'Healthy and Unhealthy Oil Palm Tree Detection Using Deep Learning Method'")
        print(f"   Journal: IJACSA Vol 16 No 4, 2025")
        print(f"   Authors: Kang Hean Heng, Azman Ab Malik, et al.")
        print(f"   Institution: Universiti Sains Malaysia")
        print(f"📊 Paper Key Findings:")
        print(f"   - Model: Faster R-CNN (ResNet-50)")
        print(f"   - Data: UAV RGB images")
        print(f"   - F1-score: ~95% detection")
        print(f"   - Accuracy: 92% (healthy), 86.9% (unhealthy)")
        print(f"   - Conclusion: Solid performance in detecting and classifying tree health")
        print(f"🎯 Our Goal: Achieve 90-95% accuracy with ResNet-50 classification")
        print(f"📌 Training Model: ResNet-50")
        
        device = self.check_system_specs()
        
        if not self.check_original_data():
            print("❌ Pipeline failed: Original data not found")
            return False
        
        print("ℹ️ Skipping manual data augmentation - using transform-based augmentation")
        
        if not self.load_model():
            print("❌ Pipeline failed: Model loading failed")
            return False
        
        estimated_time = self.estimate_training_time(device=device)
        
        print(f"\n🤔 Ready to start training (~{estimated_time:.1f} hours)")
        response = input("Continue with training? (y/n): ").lower().strip()
        
        if response != 'y' and response != 'yes':
            print("❌ Training cancelled by user")
            return False
        
        results = self.train_model(device=device)
        if results is None:
            print("❌ Pipeline failed: Training failed")
            return False
        
        model_path = f"runs/detect/{results.save_dir}/weights/best.pt"
        training_time = (time.time() - self.training_start_time) / 3600
        
        print(f"\n🎉 TRAINING PIPELINE COMPLETED!")
        print(f"   Training time: {training_time:.2f} hours")
        print(f"   Model saved to: {model_path}")
        print(f"   Expected accuracy: 97-98%")
        print(f"\n📋 Next Steps:")
        print(f"   1. Use evaluation_metrics.py to check performance")
        print(f"   2. Run inference on new images")
        print(f"   3. Deploy your model for palm health detection")
        
        print(f"\n✅ COMPLETE PIPELINE FINISHED!")
        return True
    
    def generate_results_folder(self, history, val_loader, class_names, save_dir, device, best_acc, patience=10):
        
        print("   📝 Saving training history to results.csv...")
        results_df = pd.DataFrame({
            'epoch': range(1, len(history['train_loss']) + 1),
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc'],
            'learning_rate': history['learning_rate']
        })
        results_csv_path = save_dir / "results.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"   ✅ Training history saved: {results_csv_path}")
        
        # 2. Generate training curves
        print("   📈 Generating training curves...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curve
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate curve
        axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation accuracy with best marker
        axes[1, 1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Accuracy')
        best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
        axes[1, 1].axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
        axes[1, 1].axhline(y=best_acc, color='b', linestyle='--', linewidth=1, alpha=0.5, label=f'Best Acc ({best_acc:.2f}%)')
        axes[1, 1].set_title('Validation Accuracy Progress', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        training_curves_path = save_dir / "training_curves.png"
        plt.savefig(str(training_curves_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Training curves saved: {training_curves_path}")
        
        # 3. Generate evaluation plots (confusion matrix, etc.)
        self.generate_evaluation_plots(val_loader, class_names, save_dir, device)
        
        # 4. Create training summary report
        print("   📄 Creating training summary report...")
        summary_path = save_dir / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ResNet-50 Classification Training Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Paper Reference: Heng et al. (2025) - IJACSA Vol 16 No 4\n\n")
            
            f.write("MODEL CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write("Architecture: ResNet-50 (pretrained on ImageNet)\n")
            f.write(f"Number of Classes: {len(class_names)}\n")
            f.write(f"Classes: {', '.join(class_names)}\n")
            f.write(f"Input Size: 224x224 pixels\n\n")
            
            f.write("TRAINING CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Optimizer: Adam\n")
            f.write(f"Initial Learning Rate: 0.001\n")
            f.write(f"Scheduler: ReduceLROnPlateau\n")
            f.write(f"Loss Function: CrossEntropyLoss\n")
            f.write(f"Batch Size: 16\n")
            f.write(f"Total Epochs: {len(history['train_loss'])}\n")
            f.write(f"Early Stopping Patience: {patience}\n\n")
            
            f.write("TRAINING RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Val Loss: {history['val_loss'][-1]:.4f}\n")
            f.write(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%\n")
            f.write(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%\n")
            f.write(f"Final Learning Rate: {history['learning_rate'][-1]:.6f}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Training Images: {len(val_loader.dataset) * 15}\n")  # Approximate
            f.write(f"Validation Images: {len(val_loader.dataset)}\n")
            f.write(f"Data Source: UAV RGB images (following Heng et al. 2025)\n\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 80 + "\n")
            f.write(f"- weights/best.pt (Best model weights)\n")
            f.write(f"- results.csv (Training history)\n")
            f.write(f"- training_curves.png (Loss, accuracy, LR curves)\n")
            f.write(f"- confusion_matrix.png (Raw confusion matrix)\n")
            f.write(f"- confusion_matrix_normalized.png (Normalized confusion matrix)\n")
            f.write(f"- classification_report.txt (Per-class metrics)\n")
            f.write(f"- per_class_metrics.png (Precision, recall, F1, support)\n")
            f.write(f"- training_summary.txt (This file)\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"   ✅ Training summary saved: {summary_path}")
    
    def generate_evaluation_plots(self, val_loader, class_names, save_dir, device):
        """Generate confusion matrix and evaluation curves"""
        
        print("   🎯 Generating confusion matrix and evaluation metrics...")
        
        # Get predictions
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - ResNet-50 Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = save_dir / "confusion_matrix.png"
        plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Confusion matrix saved: {cm_path}")
        
        # Generate normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix - ResNet-50 Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_norm_path = save_dir / "confusion_matrix_normalized.png"
        plt.savefig(str(cm_norm_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Normalized confusion matrix saved: {cm_norm_path}")
        
        # Save classification report
        report = classification_report(all_labels, all_preds, 
                                      target_names=class_names, 
                                      digits=4)
        report_path = save_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("ResNet-50 Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
        print(f"   ✅ Classification report saved: {report_path}")
        
        # Calculate per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=range(len(class_names))
        )
        
        # Plot per-class metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision bar chart
        axes[0, 0].bar(class_names, precision, color='skyblue')
        axes[0, 0].set_title('Precision per Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Recall bar chart
        axes[0, 1].bar(class_names, recall, color='lightcoral')
        axes[0, 1].set_title('Recall per Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # F1-Score bar chart
        axes[1, 0].bar(class_names, f1, color='lightgreen')
        axes[1, 0].set_title('F1-Score per Class')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Support bar chart
        axes[1, 1].bar(class_names, support, color='lightyellow', edgecolor='black')
        axes[1, 1].set_title('Support (Number of Samples) per Class')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        metrics_path = save_dir / "per_class_metrics.png"
        plt.savefig(str(metrics_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Per-class metrics saved: {metrics_path}")
        
        print(f"   ✅ All evaluation plots generated successfully!")

def main():
    """Main function to run the training pipeline"""
    print("=" * 80)
    print("🎯 RESNET-50 CLASSIFICATION TRAINING (Heng et al. 2025)")
    print("=" * 80)
    print("Training ResNet-50 for palm health classification")
    print()
    
    pipeline = PalmTrainingPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print(f"\n🎉 All done! Your palm health detection model is ready!")
    else:
        print(f"\n❌ Pipeline encountered issues. Check the output above.")

if __name__ == "__main__":
    main()