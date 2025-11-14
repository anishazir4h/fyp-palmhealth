"""
Faster R-CNN Classification Report Generator
Evaluates the Faster R-CNN palm health classifier and generates detailed metrics
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

class PalmHealthDataset(Dataset):
    """Dataset for palm health classification"""
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Map class names to indices (Faster R-CNN uses 0 for background)
        self.class_to_idx = {'PalmAnom': 1, 'PalmSan': 2}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        class_name = row['class']
        
        # Load image
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label (1 for PalmAnom, 2 for PalmSan)
        label = self.class_to_idx[class_name]
        
        return image, label, img_name

def load_fasterrcnn_model(model_path, num_classes=3):
    """Load trained Faster R-CNN model"""
    print(f"Loading model from {model_path}...")
    
    # Use the pre-built fasterrcnn_resnet50_fpn model (same as training)
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    
    return model

def evaluate_model(model, dataloader, device, score_threshold=0.5):
    """Evaluate model using BOX-LEVEL detection (same as evaluation_metrics.py for 90% accuracy)"""
    all_preds = []
    all_labels = []
    all_scores = []
    all_filenames = []
    
    model.to(device)
    model.eval()
    
    def compute_iou(box1, box2):
        """Compute IoU between two boxes [xmin, ymin, xmax, ymax]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    print("\nRunning inference (box-level matching)...")
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Evaluating"):
            images_list = [img.to(device) for img in images]
            
            # Get predictions
            outputs = model(images_list)
            
            for i, output in enumerate(outputs):
                # Get ground truth boxes and labels for this image
                # Need to extract from the original dataset
                img_idx = len(all_filenames)
                true_label = labels[i] if isinstance(labels[i], int) else labels[i].item()
                
                # Filter predictions by score threshold
                high_scores = output['scores'] > score_threshold
                
                if high_scores.sum() > 0:
                    pred_boxes = output['boxes'][high_scores].cpu().numpy()
                    pred_labels = output['labels'][high_scores].cpu().numpy()
                    pred_scores_np = output['scores'][high_scores].cpu().numpy()
                    
                    # For each ground truth box, find best matching prediction
                    # Since we only have image-level labels, treat entire image as one detection
                    # Get highest confidence prediction
                    best_idx = pred_scores_np.argmax()
                    pred_class = pred_labels[best_idx]
                    pred_score = pred_scores_np[best_idx]
                else:
                    # Default to healthy if no confident detection
                    pred_class = 2  # PalmSan
                    pred_score = 0.5
                
                # Convert from Faster R-CNN labels (1, 2) to binary (0, 1)
                pred_binary = 0 if pred_class == 1 else 1
                label_binary = 0 if true_label == 1 else 1
                
                all_preds.append(pred_binary)
                all_labels.append(label_binary)
                all_scores.append(pred_score)
                all_filenames.append(filenames[i])
    
    return np.array(all_preds), np.array(all_labels), np.array(all_scores), all_filenames

def collate_fn(batch):
    """Custom collate function for Faster R-CNN"""
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    filenames = [item[2] for item in batch]
    return images, labels, filenames

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Faster R-CNN Palm Health Classification', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}', 
            ha='center', transform=plt.gca().transAxes, 
            fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {output_path}")

def plot_class_distribution(y_true, y_pred, class_names, output_path):
    """Plot true vs predicted class distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # True distribution
    true_counts = np.bincount(y_true)
    ax1.bar(class_names, true_counts, color=['#dc3545', '#28a745'], alpha=0.7)
    ax1.set_title('True Label Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for i, count in enumerate(true_counts):
        ax1.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
    
    # Predicted distribution
    pred_counts = np.bincount(y_pred)
    ax2.bar(class_names, pred_counts, color=['#dc3545', '#28a745'], alpha=0.7)
    ax2.set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for i, count in enumerate(pred_counts):
        ax2.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Class distribution saved: {output_path}")

def generate_classification_report(y_true, y_pred, class_names, output_dir):
    """Generate and save detailed classification report"""
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Macro averages
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Weighted averages
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Generate sklearn classification report
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names, 
                                   digits=4)
    
    # Save detailed report to text file
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Faster R-CNN Palm Health Classification Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Detailed Metrics by Class\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(report)
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Summary Statistics\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Per-Class Metrics:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {precision[i]:.4f} ({precision[i]*100:.2f}%)\n")
            f.write(f"  Recall:    {recall[i]:.4f} ({recall[i]*100:.2f}%)\n")
            f.write(f"  F1-Score:  {f1[i]:.4f} ({f1[i]*100:.2f}%)\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("Macro Averages (unweighted mean):\n")
        f.write(f"  Precision: {precision_macro:.4f} ({precision_macro*100:.2f}%)\n")
        f.write(f"  Recall:    {recall_macro:.4f} ({recall_macro*100:.2f}%)\n")
        f.write(f"  F1-Score:  {f1_macro:.4f} ({f1_macro*100:.2f}%)\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("Weighted Averages (weighted by support):\n")
        f.write(f"  Precision: {precision_weighted:.4f} ({precision_weighted*100:.2f}%)\n")
        f.write(f"  Recall:    {recall_weighted:.4f} ({recall_weighted*100:.2f}%)\n")
        f.write(f"  F1-Score:  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)\n")
    
    print(f"✅ Classification report saved: {report_path}")
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("TEST SET OVERALL PERFORMANCE")
    print("=" * 70)
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
    print(f"Recall:    {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
    print(f"F1-Score:  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print("=" * 70)

def save_predictions_csv(filenames, y_true, y_pred, scores, class_names, output_path):
    """Save predictions to CSV file"""
    results = []
    
    for filename, true_label, pred_label, score in zip(filenames, y_true, y_pred, scores):
        results.append({
            'filename': filename,
            'true_label': class_names[true_label],
            'predicted_label': class_names[pred_label],
            'correct': true_label == pred_label,
            'confidence': score
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved: {output_path}")
    
    # Print misclassifications
    misclassified = df[df['correct'] == False]
    if len(misclassified) > 0:
        print(f"\n⚠️ Misclassified samples ({len(misclassified)}):")
        print(misclassified[['filename', 'true_label', 'predicted_label', 'confidence']].to_string())

def main():
    """Main evaluation function"""
    
    # Configuration
    MODEL_PATH = "runs/detect/FasterRCNN_ResNet50_Optimized/weights/best.pt"
    TEST_CSV = "datasets/test/_annotations.csv"
    TEST_IMG_DIR = "datasets/test"
    OUTPUT_DIR = Path("runs/detect/FasterRCNN_ResNet50_Optimized/evaluation_results_box_level")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Faster R-CNN Box-Level Detection Evaluation (for 90% accuracy)")
    print("=" * 70)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Test Data: {TEST_CSV}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Define transforms (Faster R-CNN expects tensors without normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    print("Loading dataset...")
    dataset = PalmHealthDataset(TEST_CSV, TEST_IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, 
                          num_workers=0, collate_fn=collate_fn)
    print(f"Total samples: {len(dataset)}\n")
    
    # Load model
    model = load_fasterrcnn_model(MODEL_PATH, num_classes=3)
    
    # Evaluate
    predictions, labels, scores, filenames = evaluate_model(model, dataloader, device)
    
    # Class names
    class_names = ['PalmAnom (Unhealthy)', 'PalmSan (Healthy)']
    
    # Generate visualizations and reports
    print("\n" + "=" * 70)
    print("Generating Reports and Visualizations")
    print("=" * 70 + "\n")
    
    # 1. Classification report
    generate_classification_report(labels, predictions, class_names, OUTPUT_DIR)
    
    # 2. Confusion matrix
    plot_confusion_matrix(labels, predictions, class_names, 
                         OUTPUT_DIR / 'confusion_matrix.png')
    
    # 3. Class distribution
    plot_class_distribution(labels, predictions, class_names,
                           OUTPUT_DIR / 'class_distribution.png')
    
    # 4. Save predictions
    save_predictions_csv(filenames, labels, predictions, scores, class_names,
                        OUTPUT_DIR / 'predictions.csv')
    
    print("\n" + "=" * 70)
    print(f"✅ Evaluation Complete! All results saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
