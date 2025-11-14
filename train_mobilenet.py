import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from pathlib import Path
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class PalmHealthDataset(Dataset):
    """Custom dataset for palm health classification"""
    
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Automatically get classes from CSV (PalmSan, PalmAnom)
        self.classes = sorted(self.df['class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Create image-label mapping
        self.image_labels = {}
        for _, row in self.df.iterrows():
            filename = row['filename']
            class_name = row['class']
            self.image_labels[filename] = self.class_to_idx[class_name]
        
        self.images = list(self.image_labels.keys())
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Get label
        label = self.image_labels[img_name]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MobileNetV2Trainer:
    """Complete training pipeline for MobileNetV2 classification"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        
        # Paths
        self.train_csv = "datasets/train_aug/_annotations.csv"
        self.train_img = "datasets/train_aug/"
        self.val_csv = "datasets/valid_aug/_annotations.csv"
        self.val_img = "datasets/valid_aug/"
        self.save_dir = "runs/detect/MobileNetV2_Classification_Augmented"
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/weights", exist_ok=True)
        
    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "=" * 80)
        print(f"🎯 {title}")
        print("=" * 80)
    
    def check_system_specs(self):
        """Display system specifications"""
        self.print_header("SYSTEM SPECIFICATIONS")
        
        print(f"💻 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"   CPU Training: Optimized for efficiency")
        
        print(f"\n📦 PyTorch Version: {torch.__version__}")
    
    def check_dataset(self):
        """Verify dataset availability"""
        self.print_header("DATASET VERIFICATION")
        
        # Check files exist
        if not os.path.exists(self.train_csv):
            raise FileNotFoundError(f"Training CSV not found: {self.train_csv}")
        if not os.path.exists(self.val_csv):
            raise FileNotFoundError(f"Validation CSV not found: {self.val_csv}")
        
        # Load and analyze
        train_df = pd.read_csv(self.train_csv)
        val_df = pd.read_csv(self.val_csv)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Training images: {len(train_df)}")
        print(f"   Validation images: {len(val_df)}")
        
        # Class distribution
        print(f"\n📈 Training Class Distribution:")
        train_dist = train_df['class'].value_counts()
        for class_name, count in train_dist.items():
            percentage = (count / len(train_df)) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\n📈 Validation Class Distribution:")
        val_dist = val_df['class'].value_counts()
        for class_name, count in val_dist.items():
            percentage = (count / len(val_df)) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        return len(train_df), len(val_df)
    
    def create_dataloaders(self, batch_size=32):
        """Create train and validation dataloaders"""
        self.print_header("DATA LOADING")
        
        print(f"🔄 Creating dataloaders with batch_size={batch_size}...")
        
        # MobileNetV2 uses ImageNet pretrained weights, so use ImageNet normalization
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = PalmHealthDataset(
            csv_file=self.train_csv,
            img_dir=self.train_img,
            transform=train_transform
        )
        
        val_dataset = PalmHealthDataset(
            csv_file=self.val_csv,
            img_dir=self.val_img,
            transform=val_transform
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✅ Dataloaders created successfully!")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
        
        return True
    
    def build_model(self):
        """Build MobileNetV2 model"""
        self.print_header("MODEL ARCHITECTURE")
        
        print(f"🏗️ Building MobileNetV2 model...")
        
        # Load pretrained MobileNetV2
        self.model = models.mobilenet_v2(pretrained=True)
        
        # Get number of input features for the classifier
        num_ftrs = self.model.classifier[1].in_features
        
        # Replace the classifier with a new one for binary classification
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 2)  # 2 classes: Healthy, Unhealthy
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model built successfully!")
        print(f"   Architecture: MobileNetV2")
        print(f"   Total parameters: {total_params / 1e6:.2f}M")
        print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"   Input size: 224x224")
        print(f"   Output classes: 2 (PalmSan, PalmAnom)")
        
        return True
    
    def setup_training(self, learning_rate=0.001):
        """Setup loss function, optimizer, and scheduler"""
        self.print_header("TRAINING SETUP")
        
        print(f"⚙️ Configuring training components...")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer - Adam for MobileNetV2
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler - reduce on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        print(f"✅ Training setup complete!")
        print(f"   Loss function: CrossEntropyLoss")
        print(f"   Optimizer: Adam")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Weight decay: 1e-4")
        print(f"   LR Scheduler: ReduceLROnPlateau (patience=5)")
        
        return True
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress update every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"   Epoch [{epoch}] Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {100 * correct / total:.2f}%")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=50):
        """Complete training loop"""
        self.print_header("MODEL TRAINING")
        
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"   Device: {self.device}")
        print(f"   Target: 78-85% accuracy")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\n📊 Epoch [{epoch}/{epochs}] Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"   Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), f"{self.save_dir}/weights/best.pt")
                print(f"   ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"{self.save_dir}/weights/epoch_{epoch}.pt")
                print(f"   💾 Checkpoint saved at epoch {epoch}")
            
            print("-" * 80)
        
        # Save final model
        torch.save(self.model.state_dict(), f"{self.save_dir}/weights/last.pt")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"   Total time: {total_time / 3600:.2f} hours")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"   Models saved to: {self.save_dir}/weights/")
        
        return True
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        self.print_header("GENERATING TRAINING CURVES")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training curves saved to: {self.save_dir}/training_curves.png")
    
    def evaluate_on_validation(self):
        """Final evaluation on validation set"""
        self.print_header("FINAL VALIDATION EVALUATION")
        
        print(f"📊 Evaluating best model on validation set...")
        
        # Load best model
        self.model.load_state_dict(torch.load(f"{self.save_dir}/weights/best.pt"))
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        print(f"\n📈 Validation Metrics:")
        print(f"   Accuracy: {accuracy * 100:.2f}%")
        print(f"   Precision: {precision * 100:.2f}%")
        print(f"   Recall: {recall * 100:.2f}%")
        print(f"   F1-Score: {f1 * 100:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Get class names from dataset
        class_names = [name for name, _ in sorted(self.train_loader.dataset.class_to_idx.items(), key=lambda x: x[1])]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix - MobileNetV2', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved to: {self.save_dir}/confusion_matrix.png")
        
        # Classification report
        report = classification_report(all_labels, all_preds, 
                                      target_names=class_names)
        print(f"\n📋 Classification Report:")
        print(report)
        
        # Save report to file
        with open(f"{self.save_dir}/classification_report.txt", 'w') as f:
            f.write("MobileNetV2 Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.2f}%\n")
            f.write(f"Final Accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"Precision: {precision * 100:.2f}%\n")
            f.write(f"Recall: {recall * 100:.2f}%\n")
            f.write(f"F1-Score: {f1 * 100:.2f}%\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(report)
        
        print(f"✅ Report saved to: {self.save_dir}/classification_report.txt")
        
        return accuracy
    
    def run_complete_pipeline(self, batch_size=32, epochs=50, learning_rate=0.001):
        """Run the complete training pipeline"""
        print("=" * 80)
        print("🎯 MOBILENETV2 PALM HEALTH CLASSIFICATION - TRAINING PIPELINE")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Goal: Lightweight efficient model for resource-constrained deployment")
        
        # Step 1: System check
        self.check_system_specs()
        
        # Step 2: Dataset verification
        train_size, val_size = self.check_dataset()
        
        # Step 3: Create dataloaders
        self.create_dataloaders(batch_size=batch_size)
        
        # Step 4: Build model
        self.build_model()
        
        # Step 5: Setup training
        self.setup_training(learning_rate=learning_rate)
        
        # Step 6: Estimate training time
        print(f"\n⏱️ TRAINING TIME ESTIMATE:")
        print(f"   Dataset size: {train_size} training images")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Estimated time: 2-3 hours on CPU")
        print(f"   Expected accuracy: 78-85%")
        
        # Ask for confirmation
        print(f"\n🤔 Ready to start training")
        response = input("Continue? (y/n): ").lower().strip()
        
        if response != 'y' and response != 'yes':
            print("❌ Training cancelled by user")
            return False
        
        # Step 7: Train model
        self.train(epochs=epochs)
        
        # Step 8: Plot training curves
        self.plot_training_curves()
        
        # Step 9: Final evaluation
        final_acc = self.evaluate_on_validation()
        
        # Summary
        print(f"\n" + "=" * 80)
        print(f"🎉 TRAINING PIPELINE COMPLETED!")
        print(f"=" * 80)
        print(f"   Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"   Final Accuracy: {final_acc * 100:.2f}%")
        print(f"   Model saved to: {self.save_dir}/weights/best.pt")
        print(f"\n📋 Next Steps:")
        print(f"   1. Use evaluation_metrics.py to compare with other models")
        print(f"   2. Test on new palm images")
        print(f"   3. Deploy for edge/mobile applications")
        print(f"\n✅ MobileNetV2 training complete!")
        
        return True


def main():
    """Main function to run the training pipeline"""
    trainer = MobileNetV2Trainer()
    success = trainer.run_complete_pipeline(
        batch_size=32,
        epochs=50,
        learning_rate=0.001
    )
    
    if success:
        print(f"\n🎉 MobileNetV2 training successful! Ready for model comparison.")
    else:
        print(f"\n❌ Training encountered issues. Check the output above.")


if __name__ == "__main__":
    main()
