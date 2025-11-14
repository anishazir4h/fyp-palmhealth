"""
ResNet-50 Classification Report Generator
Evaluates the ResNet-50 palm health classifier and generates detailed metrics
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
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
        
        # Map class names to indices
        self.class_to_idx = {'PalmAnom': 0, 'PalmSan': 1}
        
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
        
        # Get label
        label = self.class_to_idx[class_name]
        
        return image, label, img_name

def load_resnet50_model(model_path, num_classes=2):
    """Load trained ResNet-50 model"""
    print(f"Loading model from {model_path}...")
    
    # Create ResNet-50 architecture
    model = models.resnet50(pretrained=False)
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
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

def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions and ground truth"""
    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []
    
    model.to(device)
    model.eval()
    
    print("\nRunning inference...")
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_filenames.extend(filenames)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_filenames

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - ResNet-50 Palm Health Classification', 
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
        f.write("ResNet-50 Palm Health Classification Report\n")
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

def save_predictions_csv(filenames, y_true, y_pred, probabilities, class_names, output_path):
    """Save predictions to CSV file"""
    results = []
    
    for filename, true_label, pred_label, probs in zip(filenames, y_true, y_pred, probabilities):
        results.append({
            'filename': filename,
            'true_label': class_names[true_label],
            'predicted_label': class_names[pred_label],
            'correct': true_label == pred_label,
            'prob_PalmAnom': probs[0],
            'prob_PalmSan': probs[1],
            'confidence': max(probs)
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
    MODEL_PATH = "runs/detect/ResNet50_Classification_Augmented/weights/best.pt"
    TEST_CSV = "datasets/test/_annotations.csv"  # or "datasets/valid/_annotations.csv"
    TEST_IMG_DIR = "datasets/test"  # or "datasets/valid"
    OUTPUT_DIR = Path("runs/detect/ResNet50_Classification_Augmented/evaluation_results")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("ResNet-50 Classification Evaluation")
    print("=" * 70)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Test Data: {TEST_CSV}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("Loading dataset...")
    dataset = PalmHealthDataset(TEST_CSV, TEST_IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Total samples: {len(dataset)}\n")
    
    # Load model
    model = load_resnet50_model(MODEL_PATH, num_classes=2)
    
    # Evaluate
    predictions, labels, probabilities, filenames = evaluate_model(model, dataloader, device)
    
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
    save_predictions_csv(filenames, labels, predictions, probabilities, class_names,
                        OUTPUT_DIR / 'predictions.csv')
    
    print("\n" + "=" * 70)
    print(f"✅ Evaluation Complete! All results saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
