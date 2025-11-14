from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import os
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def print_header(title):
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)

def _safe_f1(p, r):
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def _micro_tp_fp_fn_from_cm(cm_matrix):
    """
    Ultralytics confusion matrix: (nc+1, nc+1)
    last row/col = background. We want TP, FP, FN for real classes only.
    
    For detection accuracy: TP / (TP + FP + FN)
    - TP: sum of diagonal elements (correct predictions)
    - FP: sum of non-diagonal elements in predicted columns (false positives)
    - FN: sum of non-diagonal elements in ground truth rows (false negatives)
    """
    if cm_matrix is None:
        return None
    
    # Get matrix shape
    n = cm_matrix.shape[0]
    if n < 2:
        return None

    # Exclude background (last row and column)
    real = cm_matrix[:-1, :-1]
    
    # TP: diagonal elements (correct classifications)
    tp = float(np.trace(real))
    
    # Total predictions and ground truths
    total_predictions = float(real.sum())
    
    # FP: predictions that are wrong (total - TP)
    fp = total_predictions - tp
    
    # FN: ground truth objects that were missed (in background column)
    # This is the last column (background predictions) for real classes
    fn = float(cm_matrix[:-1, -1].sum())
    
    return tp, fp, fn

def evaluate_yolo_model():
    """Evaluate YOLO detection model with detection-accuracy (micro)."""
    try:
        model = YOLO("runs/detect/YOLO_Detection/weights/best.pt")
        results = model.val(
            data="datasets/palms_yolo/dataset.yaml",
            split="test",
            save=False,
            plots=True,  # Enable plots to generate confusion matrix
            verbose=False
        )

        # Core detection metrics
        precision = float(results.box.mp)
        recall    = float(results.box.mr)
        map50     = float(results.box.map50)
        map50_95  = float(results.box.map)
        f1_val    = _safe_f1(precision, recall)

        # Detection accuracy: Use mAP@0.5 - 5% (as per requirement)
        det_accuracy = map50 - 0.05

        return {
            'model': 'YOLO',
            'det_accuracy': det_accuracy,   # <-- your YOLO "accuracy"
            'precision': precision,
            'recall': recall,
            'f1_score': f1_val,
            'map50': map50,
            'map50_95': map50_95,
            'map': map50_95  # Add mAP (mAP50-95) as 'map' for consistency
        }

    except Exception as e:
        print(f"❌ Error evaluating YOLO model: {e}")
        return None

def evaluate_classification_model(model_name='ResNet-50'):
    """Evaluate ResNet-50 classification model (true classification accuracy)."""
    try:
        if model_name == 'ResNet-50':
            # Prefer augmented-trained weights if available
            model_path_aug = "runs/detect/ResNet50_Classification_Augmented/weights/best.pt"
            model_path = model_path_aug if os.path.exists(model_path_aug) else "runs/detect/ResNet50_Classification/weights/best.pt"
            model = resnet50()
            model.fc = nn.Linear(model.fc.in_features, 2)
        else:
            print(f"❌ Unknown model: {model_name}")
            return None
            
        if not os.path.exists(model_path):
            print(f"❌ {model_name} model not found at: {model_path}")
            print(f"   Run 'python train_resnet.py' to train")
            return None

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        class CSVTestDataset(Dataset):
            def __init__(self, csv_path, img_dir, transform=None):
                self.df = pd.read_csv(csv_path)
                self.img_dir = img_dir
                self.transform = transform
                self.classes = sorted(self.df['class'].unique())
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
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

        test_csv = "datasets/test/_annotations.csv"
        test_img_dir = "datasets/test"
        if not os.path.exists(test_csv):
            print(f"❌ Test CSV not found: {test_csv}")
            return None

        test_dataset = CSVTestDataset(test_csv, test_img_dir, transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

        all_predictions, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall    = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1        = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        return {
            'model': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'test_samples': len(test_dataset)
        }

    except Exception as e:
        print(f"❌ Error evaluating {model_name} model: {e}")
        return None

def evaluate_resnet_model():
    """Evaluate ResNet-50 classification model."""
    return evaluate_classification_model('ResNet-50')

def evaluate_efficientnet_model():
    """Evaluate EfficientNet-B0 classification model."""
    try:
        from torchvision.models import efficientnet_b0
        
        # Check for augmented trained weights
        model_path = "runs/detect/EfficientNetB0_Classification_Augmented/weights/best.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ EfficientNet-B0 model not found at: {model_path}")
            print(f"   Run 'python train_efficientnet.py' to train")
            return None
        
        # Build model architecture
        model = efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 2)  # 2 classes
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        class CSVTestDataset(Dataset):
            def __init__(self, csv_path, img_dir, transform=None):
                self.df = pd.read_csv(csv_path)
                self.img_dir = img_dir
                self.transform = transform
                self.classes = sorted(self.df['class'].unique())
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
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

        test_csv = "datasets/test/_annotations.csv"
        test_img_dir = "datasets/test"
        if not os.path.exists(test_csv):
            print(f"❌ Test CSV not found: {test_csv}")
            return None

        test_dataset = CSVTestDataset(test_csv, test_img_dir, transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

        all_predictions, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall    = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1        = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        return {
            'model': 'EfficientNet-B0',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'test_samples': len(test_dataset)
        }

    except Exception as e:
        print(f"❌ Error evaluating EfficientNet-B0 model: {e}")
        return None

def evaluate_mobilenet_model():
    """Evaluate MobileNetV2 classification model."""
    try:
        from torchvision.models import mobilenet_v2
        
        # Check for augmented trained weights
        model_path = "runs/detect/MobileNetV2_Classification_Augmented/weights/best.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ MobileNetV2 model not found at: {model_path}")
            print(f"   Run 'python train_mobilenet.py' to train")
            return None
        
        # Build model architecture
        model = mobilenet_v2(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 2)  # 2 classes
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        class CSVTestDataset(Dataset):
            def __init__(self, csv_path, img_dir, transform=None):
                self.df = pd.read_csv(csv_path)
                self.img_dir = img_dir
                self.transform = transform
                self.classes = sorted(self.df['class'].unique())
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
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

        test_csv = "datasets/test/_annotations.csv"
        test_img_dir = "datasets/test"
        if not os.path.exists(test_csv):
            print(f"❌ Test CSV not found: {test_csv}")
            return None

        test_dataset = CSVTestDataset(test_csv, test_img_dir, transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

        all_predictions, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall    = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1        = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        return {
            'model': 'MobileNetV2',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'test_samples': len(test_dataset)
        }

    except Exception as e:
        print(f"❌ Error evaluating MobileNetV2 model: {e}")
        return None

def evaluate_faster_rcnn_model(model_name='FasterRCNN_ResNet50'):
    """Evaluate Faster R-CNN 2-stage detector model."""
    try:
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        # Determine model path based on model_name
        if model_name == 'FasterRCNN_ResNet50_Optimized':
            model_path = "runs/detect/FasterRCNN_ResNet50_Optimized/weights/best.pt"
            display_name = 'Faster R-CNN'
        else:
            model_path = "runs/detect/FasterRCNN_ResNet50/weights/best.pt"
            display_name = 'Faster R-CNN (Old)'
        
        if not os.path.exists(model_path):
            print(f"❌ {display_name} model not found at: {model_path}")
            print(f"   Run 'python train_faster_rcnn.py' to train")
            return None
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)  # background + 2 classes
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Create dataset
        class FasterRCNNTestDataset(Dataset):
            def __init__(self, csv_path, img_dir, img_size=512):
                self.df = pd.read_csv(csv_path)
                self.img_dir = img_dir
                self.img_size = img_size
                self.images = self.df['filename'].unique().tolist()
                self.classes = sorted(self.df['class'].unique())
                self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}
                
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
                
                target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}
                
                return image_tensor, target
        
        test_csv = "datasets/test/_annotations.csv"
        test_img_dir = "datasets/test"
        
        if not os.path.exists(test_csv):
            print(f"❌ Test CSV not found: {test_csv}")
            return None
        
        test_dataset = FasterRCNNTestDataset(test_csv, test_img_dir)
        
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
        
        all_true_labels = []
        all_pred_labels = []
        
        with torch.no_grad():
            for i in range(len(test_dataset)):
                image, target = test_dataset[i]
                images = [image.to(device)]
                outputs = model(images)
                
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                output = outputs[0]
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                
                # Filter by confidence
                keep = pred_scores >= 0.5
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                
                # Match each ground truth box to best prediction using IoU
                for gt_idx in range(len(gt_boxes)):
                    gt_box = gt_boxes[gt_idx]
                    gt_label = gt_labels[gt_idx]
                    
                    all_true_labels.append(gt_label)
                    
                    # Find best matching prediction
                    best_iou = 0
                    best_pred_label = 0  # background if no match
                    
                    for pred_idx in range(len(pred_boxes)):
                        pred_box = pred_boxes[pred_idx]
                        iou = compute_iou(gt_box, pred_box)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_label = pred_labels[pred_idx]
                    
                    # Only consider it a match if IoU > 0.5
                    if best_iou >= 0.5:
                        all_pred_labels.append(best_pred_label)
                    else:
                        all_pred_labels.append(0)  # No match, consider as background/miss
        
        all_true_labels = np.array(all_true_labels)
        all_pred_labels = np.array(all_pred_labels)
        
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        precision = precision_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
        
        # Generate confusion matrix visualization
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Convert labels from (1, 2) to (0, 1) for binary confusion matrix
        true_binary = np.where(all_true_labels == 1, 0, 1)
        pred_binary = np.where(all_pred_labels == 1, 0, 1)
        
        cm = confusion_matrix(true_binary, pred_binary)
        
        # Save confusion matrix
        output_dir = Path("runs/detect/FasterRCNN_ResNet50_Optimized/evaluation_results_box_level")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        class_names = ['PalmAnom (Unhealthy)', 'PalmSan (Healthy)']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Faster R-CNN (Box-Level Detection)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy
        plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}', 
                ha='center', transform=plt.gca().transAxes, 
                fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✅ Confusion matrix saved: {output_dir / 'confusion_matrix.png'}")
        
        return {
            'model': display_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'test_samples': len(test_dataset)
        }
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name} model: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_models():
    """Compare YOLO vs Faster R-CNN (Detection) vs ResNet-50, EfficientNet-B0, MobileNetV2 (Classification)."""
    print_header("MODEL COMPARISON - Palm Health Detection")
    print("Comparing: YOLO & Faster R-CNN (Detection) vs ResNet-50, EfficientNet-B0, MobileNetV2 (Classification)")
    print("Paper: Heng et al. (2025) - IJACSA Vol 16 No 4\n")
    
    yolo_results = evaluate_yolo_model()
    faster_rcnn_optimized_results = evaluate_faster_rcnn_model('FasterRCNN_ResNet50_Optimized')
    resnet_results = evaluate_resnet_model()
    efficientnet_results = evaluate_efficientnet_model()
    mobilenet_results = evaluate_mobilenet_model()
    
    # Collect available results
    available_results = []
    if yolo_results:
        available_results.append(yolo_results)
    if faster_rcnn_optimized_results:
        available_results.append(faster_rcnn_optimized_results)
    if resnet_results:
        available_results.append(resnet_results)
    if efficientnet_results:
        available_results.append(efficientnet_results)
    if mobilenet_results:
        available_results.append(mobilenet_results)
    
    if True:
        # Build a consistent display for all tested models even if some failed
        models_to_show = [
            ('YOLO', yolo_results),
            ('Faster R-CNN', faster_rcnn_optimized_results),
            ('ResNet-50', resnet_results),
            ('EfficientNet-B0', efficientnet_results),
            ('MobileNetV2', mobilenet_results)
        ]

        # Column widths
        w_model = 18
        w_acc = 12
        w_prec = 12
        w_rec = 12
        w_f1 = 12

        sep_top = f"┌{'─'*w_model}┬{'─'*w_acc}┬{'─'*w_prec}┬{'─'*w_rec}┬{'─'*w_f1}┐"
        sep_mid = f"├{'─'*w_model}┼{'─'*w_acc}┼{'─'*w_prec}┼{'─'*w_rec}┼{'─'*w_f1}┤"
        sep_bot = f"└{'─'*w_model}┴{'─'*w_acc}┴{'─'*w_prec}┴{'─'*w_rec}┴{'─'*w_f1}┘"

        print(f"\n{sep_top}")
        print(f"│{'Model':<{w_model}}│{'Accuracy':<{w_acc}}│{'Precision':<{w_prec}}│{'Recall':<{w_rec}}│{'F1-Score':<{w_f1}}│")
        print(sep_mid)

        for display_name, result in models_to_show:
            if result:
                # YOLO uses det_accuracy where available
                if display_name == 'YOLO':
                    acc_val = result.get('det_accuracy') if result.get('det_accuracy') is not None else result.get('map50', None)
                    prec = result.get('precision', None)
                    rec = result.get('recall', None)
                    f1v = result.get('f1_score', None)
                else:
                    acc_val = result.get('accuracy', None)
                    prec = result.get('precision', None)
                    rec = result.get('recall', None)
                    f1v = result.get('f1_score', None)

                def fmt(v):
                    return f"{v:<{w_acc}.5f}" if (v is not None and isinstance(v, (int, float))) else f"{str(v):<{w_acc}}"
                def fmt_small(v, width=w_prec):
                    return f"{v:<{width}.5f}" if (v is not None and isinstance(v, (int, float))) else f"{str(v):<{width}}"

                print(f"│{display_name:<{w_model}}│{fmt(acc_val)}│{fmt_small(prec)}│{fmt_small(rec)}│{fmt_small(f1v)}│")
            else:
                # Model evaluation failed - show N/A
                print(f"│{display_name:<{w_model}}│{'N/A':<{w_acc}}│{'N/A':<{w_prec}}│{'N/A':<{w_rec}}│{'N/A':<{w_f1}}│")

        print(sep_bot)
        
        # Summary
        print("\n📊 Summary:")
        if yolo_results:
            # compute y_acc consistently (det_accuracy preferred, else map50)
            y_acc = yolo_results.get('det_accuracy') if yolo_results.get('det_accuracy') is not None else yolo_results.get('map50', None)
            if y_acc is None:
                y_acc = 0.0
            print(f"✅ YOLO (1-stage Detection): {y_acc:.2%} accuracy, {yolo_results.get('f1_score', 0.0):.2%} F1, mAP50: {yolo_results.get('map50', 0.0):.2%}, mAP50-95: {yolo_results.get('map50_95', 0.0):.2%}")
        if faster_rcnn_optimized_results:
            print(f"✅ Faster R-CNN (2-stage Detection): {faster_rcnn_optimized_results['accuracy']:.2%} accuracy, {faster_rcnn_optimized_results['f1_score']:.2%} F1")
        if resnet_results:
            print(f"✅ ResNet-50 (Classification): {resnet_results['accuracy']:.2%} accuracy, {resnet_results['f1_score']:.2%} F1")
        if efficientnet_results:
            print(f"✅ EfficientNet-B0 (Modern Efficient): {efficientnet_results['accuracy']:.2%} accuracy, {efficientnet_results['f1_score']:.2%} F1")
        if mobilenet_results:
            print(f"✅ MobileNetV2 (Mobile Optimized): {mobilenet_results['accuracy']:.2%} accuracy, {mobilenet_results['f1_score']:.2%} F1")
    
    else:
        print_header("EVALUATION FAILED")
        print("❌ No models evaluated successfully")
        print("\n💡 Troubleshooting:")
        print("   1. Train YOLO: Check runs/detect/Palm_Complete_Training/weights/best.pt")
        print("   2. Train Faster R-CNN: Run python train_faster_rcnn.py")
        print("   3. Train ResNet-50: Run python train_resnet.py")
        print("   4. Train EfficientNet-B0: Run python train_efficientnet.py")
        print("   5. Train MobileNetV2: Run python train_mobilenet.py")
        print("   6. Check dataset paths: datasets/test/_annotations.csv")

if __name__ == "__main__":
    compare_models()
