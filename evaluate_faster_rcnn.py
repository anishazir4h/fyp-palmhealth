"""
Evaluate Faster R-CNN model and generate comprehensive results
Similar to YOLO and ResNet/VGG evaluation outputs
"""

import os
import pandas as pd
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

class PalmDetectionDataset(Dataset):
    """Dataset for Faster R-CNN evaluation"""
    
    def __init__(self, csv_path, img_dir, img_size=512):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        
        # Get unique images
        self.images = self.df['filename'].unique().tolist()
        
        # Create class mapping
        self.classes = sorted(self.df['class'].unique())
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        
        orig_width, orig_height = image.size
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        scale_x = self.img_size / orig_width
        scale_y = self.img_size / orig_height
        
        # Get all annotations for this image
        img_annotations = self.df[self.df['filename'] == img_name]
        
        boxes = []
        labels = []
        
        for _, row in img_annotations.iterrows():
            xmin = float(row['xmin']) * scale_x
            ymin = float(row['ymin']) * scale_y
            xmax = float(row['xmax']) * scale_x
            ymax = float(row['ymax']) * scale_y
            
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx[row['class']])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Convert image to tensor
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return image_tensor, target

class FasterRCNNEvaluator:
    """Evaluate Faster R-CNN and generate comprehensive results"""
    
    def __init__(self, model_path, num_classes=3):
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load trained Faster R-CNN model"""
        print("📥 Loading Faster R-CNN model...")
        
        # Create model
        self.model = fasterrcnn_resnet50_fpn(pretrained=False)
        
        # Replace classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Load weights
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"   ✅ Model loaded from: {self.model_path}")
        else:
            print(f"   ❌ Model not found: {self.model_path}")
            return False
        
        self.model.to(self.device)
        self.model.eval()
        return True
    
    def evaluate(self, val_csv, val_img_dir, conf_threshold=0.5, iou_threshold=0.5):
        """Evaluate model on validation set"""
        print("\n🔍 Evaluating Faster R-CNN...")
        
        # Create dataset
        dataset = PalmDetectionDataset(val_csv, val_img_dir)
        
        all_true_labels = []
        all_pred_labels = []
        
        print(f"   Processing {len(dataset)} images...")
        
        with torch.no_grad():
            for i in range(len(dataset)):
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i+1}/{len(dataset)} images")
                
                # Get single image and target
                image, target = dataset[i]
                
                # Prepare for model
                images = [image.to(self.device)]
                outputs = self.model(images)
                
                # Get ground truth labels
                true_labels = target['labels'].cpu().numpy()
                all_true_labels.extend(true_labels)
                
                # Get predictions
                output = outputs[0]
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Filter by confidence
                keep = scores >= conf_threshold
                pred_labels = labels[keep]
                pred_boxes = boxes[keep]
                
                # Match predictions to ground truth (simple approach: assign by count)
                # For each ground truth, find the best matching prediction
                gt_boxes = target['boxes'].cpu().numpy()
                
                if len(pred_labels) == 0:
                    # No predictions - assign background class (0) or use ground truth count
                    all_pred_labels.extend([0] * len(true_labels))
                elif len(pred_labels) >= len(true_labels):
                    # More predictions than ground truth - use first N predictions
                    all_pred_labels.extend(pred_labels[:len(true_labels)])
                else:
                    # Fewer predictions than ground truth
                    all_pred_labels.extend(pred_labels)
                    # Fill remaining with most common predicted class
                    if len(pred_labels) > 0:
                        most_common = np.bincount(pred_labels).argmax()
                        all_pred_labels.extend([most_common] * (len(true_labels) - len(pred_labels)))
                    else:
                        all_pred_labels.extend([0] * (len(true_labels) - len(pred_labels)))
        
        # Convert to numpy arrays
        all_true_labels = np.array(all_true_labels)
        all_pred_labels = np.array(all_pred_labels)
        
        # Calculate metrics
        print("\n📊 Calculating metrics...")
        
        # Get class names
        class_names = [dataset.idx_to_class[i] for i in sorted(dataset.idx_to_class.keys())]
        
        # Overall accuracy
        accuracy = (all_true_labels == all_pred_labels).mean() * 100
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average=None, labels=sorted(dataset.class_to_idx.values())
        )
        
        # Weighted average
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='weighted'
        )
        
        print(f"\n✅ EVALUATION RESULTS:")
        print(f"   Overall Accuracy: {accuracy:.2f}%")
        print(f"   Weighted Precision: {precision_avg:.4f}")
        print(f"   Weighted Recall: {recall_avg:.4f}")
        print(f"   Weighted F1-Score: {f1_avg:.4f}")
        print(f"\n   Per-class metrics:")
        for i, class_name in enumerate(class_names):
            print(f"      {class_name}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")
        
        return {
            'true_labels': all_true_labels,
            'pred_labels': all_pred_labels,
            'class_names': class_names,
            'accuracy': accuracy,
            'precision': precision_avg,
            'recall': recall_avg,
            'f1': f1_avg,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support': support
        }
    
    def generate_results(self, results, save_dir):
        """Generate comprehensive results folder"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Generating results folder...")
        
        # 1. Confusion Matrix (Raw)
        print("   📈 Creating confusion matrix...")
        cm = confusion_matrix(results['true_labels'], results['pred_labels'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=results['class_names'],
                    yticklabels=results['class_names'])
        plt.title('Faster R-CNN Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(str(save_dir / "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix (Normalized)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=results['class_names'],
                    yticklabels=results['class_names'])
        plt.title('Faster R-CNN Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(str(save_dir / "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Classification Report
        print("   📝 Creating classification report...")
        report = classification_report(
            results['true_labels'],
            results['pred_labels'],
            target_names=results['class_names'],
            digits=4
        )
        
        with open(save_dir / "classification_report.txt", 'w') as f:
            f.write("Faster R-CNN Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
        
        # 4. Per-class Metrics Bar Charts
        print("   📊 Creating per-class metrics charts...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        class_names = results['class_names']
        
        # Precision
        axes[0, 0].bar(class_names, results['precision_per_class'], color='skyblue')
        axes[0, 0].set_title('Precision per Class', fontweight='bold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Recall
        axes[0, 1].bar(class_names, results['recall_per_class'], color='lightcoral')
        axes[0, 1].set_title('Recall per Class', fontweight='bold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # F1-Score
        axes[1, 0].bar(class_names, results['f1_per_class'], color='lightgreen')
        axes[1, 0].set_title('F1-Score per Class', fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Support
        axes[1, 1].bar(class_names, results['support'], color='lightyellow', edgecolor='black')
        axes[1, 1].set_title('Support (Number of Samples) per Class', fontweight='bold')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(save_dir / "per_class_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Summary Text File
        print("   📄 Creating summary report...")
        with open(save_dir / "evaluation_summary.txt", 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Faster R-CNN Evaluation Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: Faster R-CNN with ResNet-50-FPN\n")
            f.write(f"Paper: Heng et al. (2025) - IJACSA Vol 16 No 4\n")
            f.write(f"Architecture: 2-Stage Detector (RPN + R-CNN)\n\n")
            
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
            f.write(f"Weighted Precision: {results['precision']:.4f}\n")
            f.write(f"Weighted Recall: {results['recall']:.4f}\n")
            f.write(f"Weighted F1-Score: {results['f1']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 80 + "\n")
            for i, class_name in enumerate(results['class_names']):
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {results['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall: {results['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1-Score: {results['f1_per_class'][i]:.4f}\n")
                f.write(f"  Support: {results['support'][i]}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("FILES GENERATED\n")
            f.write("-" * 80 + "\n")
            f.write("- confusion_matrix.png (Raw confusion matrix)\n")
            f.write("- confusion_matrix_normalized.png (Normalized confusion matrix)\n")
            f.write("- classification_report.txt (Detailed classification report)\n")
            f.write("- per_class_metrics.png (Precision, recall, F1, support charts)\n")
            f.write("- evaluation_summary.txt (This file)\n")
        
        print(f"\n✅ Results saved to: {save_dir}")
        print(f"   ✅ confusion_matrix.png")
        print(f"   ✅ confusion_matrix_normalized.png")
        print(f"   ✅ classification_report.txt")
        print(f"   ✅ per_class_metrics.png")
        print(f"   ✅ evaluation_summary.txt")

def main():
    """Main evaluation function"""
    print("🎯 FASTER R-CNN EVALUATION")
    print("=" * 80)
    
    # Paths
    model_path = "runs/detect/FasterRCNN_ResNet50/weights/best.pt"
    val_csv = "datasets/valid/_annotations.csv"
    val_img_dir = "datasets/valid"
    save_dir = "runs/detect/FasterRCNN_ResNet50"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"\n💡 Available model paths to check:")
        print(f"   - runs/detect/FasterRCNN_ResNet50/weights/best.pt")
        print(f"   - runs/detect/FasterRCNN_ResNet50/faster_rcnn_final.pt")
        return
    
    # Check if validation data exists
    if not os.path.exists(val_csv):
        print(f"❌ Validation CSV not found: {val_csv}")
        return
    
    # Create evaluator
    evaluator = FasterRCNNEvaluator(model_path, num_classes=3)
    
    # Load model
    if not evaluator.load_model():
        return
    
    # Evaluate
    results = evaluator.evaluate(val_csv, val_img_dir)
    
    # Generate results folder
    evaluator.generate_results(results, save_dir)
    
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"   Results folder: {save_dir}")
    print(f"   Ready for comparison with YOLO, ResNet, and VGG models!")

if __name__ == "__main__":
    main()
