"""
Faster R-CNN Model Evaluation & Visualization Generator
Generates confusion matrix, accuracy graphs, precision-recall curves, and other metrics
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os
from pathlib import Path
import json
from tqdm import tqdm

class FasterRCNNEvaluator:
    def __init__(self, model_path, test_data_path, output_dir):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained Faster R-CNN model (.pt file)
            test_data_path: Path to test/validation dataset
            output_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Class names
        self.class_names = ['Background', 'PalmAnom', 'PalmSan']  # 0: Background, 1: Unhealthy, 2: Healthy
        
        # Storage for predictions
        self.all_predictions = []
        self.all_labels = []
        self.all_scores = []
        
    def _load_model(self):
        """Load trained Faster R-CNN model"""
        print(f"Loading model from {self.model_path}...")
        
        # Create model architecture
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        num_classes = 3  # Background + 2 classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully!")
        return model
    
    def _get_image_files(self):
        """Get all image files from test directory"""
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = []
        
        test_path = Path(self.test_data_path)
        
        # Check for common test directory structures
        possible_paths = [
            test_path / 'images',
            test_path / 'test' / 'images',
            test_path / 'valid' / 'images',
            test_path,
        ]
        
        for path in possible_paths:
            if path.exists():
                for ext in image_extensions:
                    image_files.extend(list(path.glob(f'*{ext}')))
                if image_files:
                    break
        
        return sorted(image_files)
    
    def _get_ground_truth(self, image_path):
        """
        Get ground truth label from CSV annotations
        Returns: class_id (1 for Unhealthy, 2 for Healthy)
        """
        # Try to find annotation CSV
        csv_files = [
            Path(self.test_data_path) / '_annotations.csv',
            Path(self.test_data_path) / 'test' / '_annotations.csv',
            Path(self.test_data_path) / 'valid' / '_annotations.csv',
        ]
        
        for csv_file in csv_files:
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    filename = image_path.name
                    
                    # Find matching row
                    row = df[df['filename'] == filename]
                    if not row.empty:
                        class_name = row.iloc[0]['class']
                        
                        # Map class name to ID
                        if class_name == 'PalmSan':  # Healthy
                            return 2
                        elif class_name == 'PalmAnom':  # Unhealthy
                            return 1
                except Exception as e:
                    print(f"Error reading CSV: {e}")
                    continue
        
        # Default to class 1 if no annotation found
        return 1
    
    def _predict_image(self, image_path):
        """Run prediction on a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to 224x224 (matching training)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        pred = predictions[0]
        
        if len(pred['boxes']) > 0:
            # Get highest confidence prediction
            max_conf_idx = pred['scores'].argmax()
            predicted_class = int(pred['labels'][max_conf_idx])
            confidence = float(pred['scores'][max_conf_idx])
            
            return predicted_class, confidence
        else:
            # No detection
            return 0, 0.0
    
    def evaluate(self):
        """Run evaluation on test dataset"""
        print("\n" + "="*60)
        print("Starting Faster R-CNN Evaluation")
        print("="*60 + "\n")
        
        # Get image files
        image_files = self._get_image_files()
        print(f"Found {len(image_files)} test images\n")
        
        if len(image_files) == 0:
            print("ERROR: No test images found!")
            print(f"Searched in: {self.test_data_path}")
            return
        
        # Evaluate each image
        print("Running predictions...")
        for img_path in tqdm(image_files, desc="Evaluating"):
            # Get ground truth
            true_label = self._get_ground_truth(img_path)
            
            # Get prediction
            pred_label, confidence = self._predict_image(img_path)
            
            # Store results
            self.all_labels.append(true_label)
            self.all_predictions.append(pred_label)
            self.all_scores.append(confidence)
        
        print("\nEvaluation complete!")
        print(f"Total images evaluated: {len(self.all_labels)}")
        
        # Generate all visualizations and metrics
        self._generate_all_reports()
    
    def _generate_all_reports(self):
        """Generate all evaluation metrics and visualizations"""
        print("\n" + "="*60)
        print("Generating Evaluation Reports & Visualizations")
        print("="*60 + "\n")
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix()
        
        # 2. Classification Report
        self._generate_classification_report()
        
        # 3. Accuracy over confidence threshold
        self._plot_accuracy_vs_confidence()
        
        # 4. Precision-Recall Curve
        self._plot_precision_recall_curve()
        
        # 5. ROC Curve
        self._plot_roc_curve()
        
        # 6. Class Distribution
        self._plot_class_distribution()
        
        # 7. Confidence Distribution
        self._plot_confidence_distribution()
        
        # 8. Summary Statistics
        self._generate_summary_stats()
        
        print("\n" + "="*60)
        print(f"All results saved to: {self.output_dir}")
        print("="*60)
    
    def _plot_confusion_matrix(self):
        """Generate and save confusion matrix"""
        print("📊 Generating Confusion Matrix...")
        
        # Filter out background class (0) for binary classification
        true_labels_binary = [l for l in self.all_labels if l != 0]
        pred_labels_binary = [self.all_predictions[i] for i, l in enumerate(self.all_labels) if l != 0]
        
        # Convert to binary: 1=Unhealthy, 2=Healthy -> 0=Unhealthy, 1=Healthy
        true_binary = [1 if l == 2 else 0 for l in true_labels_binary]
        pred_binary = [1 if l == 2 else 0 for l in pred_labels_binary]
        
        # Compute confusion matrix
        cm = confusion_matrix(true_binary, pred_binary)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Unhealthy', 'Healthy'],
                    yticklabels=['Unhealthy', 'Healthy'],
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Faster R-CNN Palm Health Classification', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}', 
                ha='center', transform=plt.gca().transAxes, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: confusion_matrix.png")
    
    def _generate_classification_report(self):
        """Generate detailed classification report"""
        print("📄 Generating Classification Report...")
        
        # Filter out background
        true_labels_binary = [l for l in self.all_labels if l != 0]
        pred_labels_binary = [self.all_predictions[i] for i, l in enumerate(self.all_labels) if l != 0]
        
        # Convert to binary
        true_binary = [1 if l == 2 else 0 for l in true_labels_binary]
        pred_binary = [1 if l == 2 else 0 for l in pred_labels_binary]
        
        # Generate report
        report = classification_report(true_binary, pred_binary, 
                                       target_names=['Unhealthy', 'Healthy'],
                                       digits=4)
        
        # Save to file
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("Faster R-CNN Classification Report\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        print(f"   ✅ Saved: classification_report.txt")
    
    def _plot_accuracy_vs_confidence(self):
        """Plot accuracy at different confidence thresholds"""
        print("📈 Generating Accuracy vs Confidence Threshold...")
        
        thresholds = np.arange(0.0, 1.01, 0.05)
        accuracies = []
        sample_counts = []
        
        for thresh in thresholds:
            # Filter predictions by confidence
            filtered_indices = [i for i, score in enumerate(self.all_scores) if score >= thresh]
            
            if len(filtered_indices) == 0:
                accuracies.append(0)
                sample_counts.append(0)
                continue
            
            filtered_true = [self.all_labels[i] for i in filtered_indices]
            filtered_pred = [self.all_predictions[i] for i in filtered_indices]
            
            # Calculate accuracy
            correct = sum([1 for t, p in zip(filtered_true, filtered_pred) if t == p])
            acc = correct / len(filtered_indices) if len(filtered_indices) > 0 else 0
            
            accuracies.append(acc)
            sample_counts.append(len(filtered_indices))
        
        # Plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Confidence Threshold', fontsize=12)
        ax1.set_ylabel('Accuracy', color=color, fontsize=12)
        ax1.plot(thresholds, accuracies, color=color, linewidth=2, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Number of Samples', color=color, fontsize=12)
        ax2.plot(thresholds, sample_counts, color=color, linewidth=2, marker='s', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Model Accuracy vs Confidence Threshold', fontsize=16, fontweight='bold', pad=20)
        fig.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_vs_confidence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: accuracy_vs_confidence.png")
    
    def _plot_precision_recall_curve(self):
        """Plot Precision-Recall curve"""
        print("📉 Generating Precision-Recall Curve...")
        
        # Filter out background and convert to binary
        valid_indices = [i for i, l in enumerate(self.all_labels) if l != 0]
        true_binary = [1 if self.all_labels[i] == 2 else 0 for i in valid_indices]
        scores = [self.all_scores[i] for i in valid_indices]
        
        if len(true_binary) == 0:
            print("   ⚠️ No valid samples for PR curve")
            return
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(true_binary, scores)
        ap = average_precision_score(true_binary, scores)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Faster R-CNN', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: precision_recall_curve.png")
    
    def _plot_roc_curve(self):
        """Plot ROC curve"""
        print("📊 Generating ROC Curve...")
        
        # Filter out background and convert to binary
        valid_indices = [i for i, l in enumerate(self.all_labels) if l != 0]
        true_binary = [1 if self.all_labels[i] == 2 else 0 for i in valid_indices]
        scores = [self.all_scores[i] for i in valid_indices]
        
        if len(true_binary) == 0:
            print("   ⚠️ No valid samples for ROC curve")
            return
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(true_binary, scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Faster R-CNN', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: roc_curve.png")
    
    def _plot_class_distribution(self):
        """Plot distribution of true vs predicted classes"""
        print("📊 Generating Class Distribution...")
        
        # Count classes
        true_counts = {0: 0, 1: 0, 2: 0}
        pred_counts = {0: 0, 1: 0, 2: 0}
        
        for t in self.all_labels:
            true_counts[t] = true_counts.get(t, 0) + 1
        
        for p in self.all_predictions:
            pred_counts[p] = pred_counts.get(p, 0) + 1
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        classes = ['Background', 'Unhealthy', 'Healthy']
        colors = ['gray', '#dc3545', '#28a745']
        
        # True labels
        ax1.bar(classes, [true_counts[i] for i in range(3)], color=colors, alpha=0.7)
        ax1.set_title('True Label Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Predicted labels
        ax2.bar(classes, [pred_counts[i] for i in range(3)], color=colors, alpha=0.7)
        ax2.set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: class_distribution.png")
    
    def _plot_confidence_distribution(self):
        """Plot distribution of confidence scores"""
        print("📊 Generating Confidence Distribution...")
        
        plt.figure(figsize=(12, 6))
        
        # Filter by prediction class
        healthy_scores = [self.all_scores[i] for i in range(len(self.all_scores)) 
                         if self.all_predictions[i] == 2]
        unhealthy_scores = [self.all_scores[i] for i in range(len(self.all_scores)) 
                           if self.all_predictions[i] == 1]
        
        # Plot histograms
        plt.hist(unhealthy_scores, bins=30, alpha=0.6, label='Unhealthy', color='#dc3545')
        plt.hist(healthy_scores, bins=30, alpha=0.6, label='Healthy', color='#28a745')
        
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Confidence Score Distribution by Predicted Class', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: confidence_distribution.png")
    
    def _generate_summary_stats(self):
        """Generate summary statistics"""
        print("📝 Generating Summary Statistics...")
        
        # Calculate metrics
        true_labels_binary = [l for l in self.all_labels if l != 0]
        pred_labels_binary = [self.all_predictions[i] for i, l in enumerate(self.all_labels) if l != 0]
        
        true_binary = [1 if l == 2 else 0 for l in true_labels_binary]
        pred_binary = [1 if l == 2 else 0 for l in pred_labels_binary]
        
        total_samples = len(true_binary)
        correct = sum([1 for t, p in zip(true_binary, pred_binary) if t == p])
        accuracy = correct / total_samples if total_samples > 0 else 0
        
        # Confidence stats
        avg_confidence = np.mean(self.all_scores)
        median_confidence = np.median(self.all_scores)
        std_confidence = np.std(self.all_scores)
        
        # Save to JSON
        stats = {
            "total_samples": total_samples,
            "accuracy": float(accuracy),
            "average_confidence": float(avg_confidence),
            "median_confidence": float(median_confidence),
            "std_confidence": float(std_confidence),
            "healthy_count": sum(true_binary),
            "unhealthy_count": len(true_binary) - sum(true_binary),
        }
        
        with open(self.output_dir / 'summary_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Save to text file
        with open(self.output_dir / 'summary_stats.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("Faster R-CNN Evaluation Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"Overall Accuracy: {accuracy:.2%}\n")
            f.write(f"\nConfidence Statistics:\n")
            f.write(f"  Average: {avg_confidence:.4f}\n")
            f.write(f"  Median: {median_confidence:.4f}\n")
            f.write(f"  Std Dev: {std_confidence:.4f}\n")
            f.write(f"\nClass Distribution:\n")
            f.write(f"  Healthy: {sum(true_binary)}\n")
            f.write(f"  Unhealthy: {len(true_binary) - sum(true_binary)}\n")
        
        print(f"   ✅ Saved: summary_stats.json")
        print(f"   ✅ Saved: summary_stats.txt")


def main():
    """Main evaluation function"""
    
    # Configuration
    MODEL_PATH = "runs/detect/FasterRCNN_ResNet50_Optimized/weights/best.pt"
    TEST_DATA_PATH = "datasets/test"  # or "datasets/valid"
    OUTPUT_DIR = "runs/detect/FasterRCNN_ResNet50_Optimized/evaluation_results"
    
    print("\n" + "="*60)
    print("Faster R-CNN Model Evaluation & Visualization")
    print("="*60)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Test Data: {TEST_DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Create evaluator
    evaluator = FasterRCNNEvaluator(MODEL_PATH, TEST_DATA_PATH, OUTPUT_DIR)
    
    # Run evaluation
    evaluator.evaluate()
    
    print("\n✅ Evaluation complete! Check the output directory for all results.\n")


if __name__ == "__main__":
    main()
