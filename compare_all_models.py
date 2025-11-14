"""
Comprehensive Model Comparison Report
Compares YOLO, Faster R-CNN, and ResNet-50 Classification
Generates detailed classification reports and visualizations
"""

import os
import pandas as pd
import torch
import torchvision
from torchvision.models import resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch.nn as nn
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

class ModelComparator:
    """Compare YOLO, Faster R-CNN, and ResNet-50 Classification"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def print_header(self, title):
        print("\n" + "=" * 80)
        print(f"🎯 {title}")
        print("=" * 80)
    
    def evaluate_yolo(self):
        """Evaluate YOLO detection model"""
        self.print_header("Evaluating YOLO (Palm_Complete_Training)")
        
        try:
            model = YOLO("runs/detect/Palm_Complete_Training/weights/best.pt")
            
            # Run validation
            results = model.val(
                data="datasets/palms_yolo/dataset.yaml",
                split="test",
                save=False,
                plots=False,
                verbose=False
            )
            
            # Extract metrics
            precision = float(results.box.mp)
            recall = float(results.box.mr)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            map50 = float(results.box.map50)
            
            # Get per-class metrics
            per_class_results = results.box.class_result
            
            print(f"✅ YOLO Evaluation Complete")
            print(f"   mAP@0.5: {map50:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            
            self.results['YOLO'] = {
                'model_type': '1-Stage Detection',
                'accuracy': map50,  # Use mAP@0.5 as accuracy metric
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'per_class': per_class_results
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error evaluating YOLO: {e}")
            return False
    
    def evaluate_faster_rcnn(self):
        """Evaluate Faster R-CNN model"""
        self.print_header("Evaluating Faster R-CNN (FasterRCNN_ResNet50)")
        
        try:
            # Load model
            model = fasterrcnn_resnet50_fpn(pretrained=False)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
            
            model_path = "runs/detect/FasterRCNN_ResNet50/weights/best.pt"
            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_path}")
                return False
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            # Create dataset
            class FasterRCNNDataset(Dataset):
                def __init__(self, csv_path, img_dir, img_size=512):
                    self.df = pd.read_csv(csv_path)
                    self.img_dir = img_dir
                    self.img_size = img_size
                    self.images = self.df['filename'].unique().tolist()
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
                    image_tensor = torchvision.transforms.functional.to_tensor(image)
                    
                    target = {'boxes': boxes, 'labels': labels}
                    return image_tensor, target
            
            # Load test data
            test_csv = "datasets/test/_annotations.csv"
            test_img_dir = "datasets/test"
            
            if not os.path.exists(test_csv):
                print(f"❌ Test data not found: {test_csv}")
                return False
            
            dataset = FasterRCNNDataset(test_csv, test_img_dir)
            
            all_true_labels = []
            all_pred_labels = []
            
            with torch.no_grad():
                for i in range(len(dataset)):
                    image, target = dataset[i]
                    images = [image.to(self.device)]
                    outputs = model(images)
                    
                    true_labels = target['labels'].cpu().numpy()
                    all_true_labels.extend(true_labels)
                    
                    output = outputs[0]
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    
                    # Filter by confidence
                    keep = scores >= 0.5
                    pred_labels = labels[keep]
                    
                    # Simple matching: use predictions up to ground truth count
                    if len(pred_labels) == 0:
                        all_pred_labels.extend([0] * len(true_labels))
                    elif len(pred_labels) >= len(true_labels):
                        all_pred_labels.extend(pred_labels[:len(true_labels)])
                    else:
                        all_pred_labels.extend(pred_labels)
                        most_common = np.bincount(pred_labels).argmax() if len(pred_labels) > 0 else 0
                        all_pred_labels.extend([most_common] * (len(true_labels) - len(pred_labels)))
            
            # Calculate metrics
            all_true_labels = np.array(all_true_labels)
            all_pred_labels = np.array(all_pred_labels)
            
            accuracy = accuracy_score(all_true_labels, all_pred_labels)
            precision, recall, f1, support = precision_recall_fscore_support(
                all_true_labels, all_pred_labels, average='weighted', zero_division=0
            )
            
            # Get class names
            class_names = [dataset.idx_to_class[i] for i in sorted(dataset.idx_to_class.keys())]
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                all_true_labels, all_pred_labels, average=None, labels=sorted(dataset.class_to_idx.values()), zero_division=0
            )
            
            print(f"✅ Faster R-CNN Evaluation Complete")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            
            self.results['Faster R-CNN'] = {
                'model_type': '2-Stage Detection',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_labels': all_true_labels,
                'pred_labels': all_pred_labels,
                'class_names': class_names,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class,
                'support_per_class': support_per_class
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error evaluating Faster R-CNN: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_resnet50(self):
        """Evaluate ResNet-50 Classification model"""
        self.print_header("Evaluating ResNet-50 (ResNet50_Classification)")
        
        try:
            # Load model
            model = resnet50()
            model.fc = nn.Linear(model.fc.in_features, 2)
            
            # Prefer augmented-trained weights if available, else fall back to original
            model_path_aug = "runs/detect/ResNet50_Classification_Augmented/weights/best.pt"
            model_path = model_path_aug if os.path.exists(model_path_aug) else "runs/detect/ResNet50_Classification/weights/best.pt"
            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_path}")
                return False
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            # Create dataset
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            class CSVDataset(Dataset):
                def __init__(self, csv_path, img_dir, transform=None):
                    self.df = pd.read_csv(csv_path)
                    self.img_dir = img_dir
                    self.transform = transform
                    self.classes = sorted(self.df['class'].unique())
                    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
                    self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
                    self.image_labels = {}
                    for _, row in self.df.iterrows():
                        self.image_labels[row['filename']] = self.class_to_idx[row['class']]
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
            
            # Load test data
            test_csv = "datasets/test/_annotations.csv"
            test_img_dir = "datasets/test"
            
            if not os.path.exists(test_csv):
                print(f"❌ Test data not found: {test_csv}")
                return False
            
            dataset = CSVDataset(test_csv, test_img_dir, transform)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
            
            all_true_labels = []
            all_pred_labels = []
            
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    all_pred_labels.extend(predicted.cpu().numpy())
                    all_true_labels.extend(labels.numpy())
            
            # Calculate metrics
            all_true_labels = np.array(all_true_labels)
            all_pred_labels = np.array(all_pred_labels)
            
            accuracy = accuracy_score(all_true_labels, all_pred_labels)
            precision, recall, f1, support = precision_recall_fscore_support(
                all_true_labels, all_pred_labels, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                all_true_labels, all_pred_labels, average=None, zero_division=0
            )
            
            print(f"✅ ResNet-50 Evaluation Complete")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            
            self.results['ResNet-50'] = {
                'model_type': 'Classification',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_labels': all_true_labels,
                'pred_labels': all_pred_labels,
                'class_names': dataset.classes,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class,
                'support_per_class': support_per_class
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error evaluating ResNet-50: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_comparison_table(self):
        """Generate comparison table"""
        self.print_header("MODEL COMPARISON TABLE")
        
        if len(self.results) == 0:
            print("❌ No results to compare")
            return
        
        # Print table header
        print(f"\n┌{'─'*20}┬{'─'*20}┬{'─'*15}┬{'─'*15}┬{'─'*15}┬{'─'*15}┐")
        print(f"│{'Model':<20}│{'Type':<20}│{'Accuracy':<15}│{'Precision':<15}│{'Recall':<15}│{'F1-Score':<15}│")
        print(f"├{'─'*20}┼{'─'*20}┼{'─'*15}┼{'─'*15}┼{'─'*15}┼{'─'*15}┤")
        
        for model_name, metrics in self.results.items():
            print(f"│{model_name:<20}│{metrics['model_type']:<20}│{metrics['accuracy']:<15.4f}│{metrics['precision']:<15.4f}│{metrics['recall']:<15.4f}│{metrics['f1_score']:<15.4f}│")
        
        print(f"└{'─'*20}┴{'─'*20}┴{'─'*15}┴{'─'*15}┴{'─'*15}┴{'─'*15}┘")
    
    def generate_classification_reports(self, save_dir="comparison_results"):
        """Generate detailed classification reports for each model"""
        self.print_header("GENERATING CLASSIFICATION REPORTS")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, metrics in self.results.items():
            if 'true_labels' not in metrics:
                continue
            
            print(f"\n📄 Generating report for {model_name}...")
            
            # Classification report
            report = classification_report(
                metrics['true_labels'],
                metrics['pred_labels'],
                target_names=metrics['class_names'],
                digits=4
            )
            
            report_path = save_dir / f"{model_name.replace(' ', '_')}_classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"{model_name} Classification Report\n")
                f.write("=" * 80 + "\n\n")
                f.write(report)
            
            print(f"   ✅ Saved: {report_path}")
            
            # Confusion matrix
            cm = confusion_matrix(metrics['true_labels'], metrics['pred_labels'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=metrics['class_names'],
                       yticklabels=metrics['class_names'])
            plt.title(f'{model_name} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            cm_path = save_dir / f"{model_name.replace(' ', '_')}_confusion_matrix.png"
            plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {cm_path}")
    
    def generate_comparison_charts(self, save_dir="comparison_results"):
        """Generate comparison bar charts"""
        self.print_header("GENERATING COMPARISON CHARTS")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        precisions = [self.results[m]['precision'] for m in models]
        recalls = [self.results[m]['recall'] for m in models]
        f1_scores = [self.results[m]['f1_score'] for m in models]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        # Precision
        axes[0, 1].bar(models, precisions, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_title('Precision Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(precisions):
            axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        # Recall
        axes[1, 0].bar(models, recalls, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 0].set_title('Recall Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(recalls):
            axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        # F1-Score
        axes[1, 1].bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(f1_scores):
            axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        chart_path = save_dir / "model_comparison_charts.png"
        plt.savefig(str(chart_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {chart_path}")
    
    def run_comparison(self):
        """Run complete comparison"""
        self.print_header("COMPREHENSIVE MODEL COMPARISON")
        print("Comparing: YOLO, Faster R-CNN, and ResNet-50 Classification")
        print("Paper: Heng et al. (2025) - IJACSA Vol 16 No 4\n")
        
        # Evaluate all models
        self.evaluate_yolo()
        self.evaluate_faster_rcnn()
        self.evaluate_resnet50()
        
        # Generate outputs
        self.generate_comparison_table()
        self.generate_classification_reports()
        self.generate_comparison_charts()
        
        print("\n" + "=" * 80)
        print("🎉 COMPARISON COMPLETE!")
        print("=" * 80)
        print(f"📁 Results saved to: comparison_results/")
        print(f"   ✅ Classification reports (TXT)")
        print(f"   ✅ Confusion matrices (PNG)")
        print(f"   ✅ Comparison charts (PNG)")

def main():
    comparator = ModelComparator()
    comparator.run_comparison()

if __name__ == "__main__":
    main()
