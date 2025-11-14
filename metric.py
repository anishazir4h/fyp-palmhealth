from ultralytics import YOLO
import numpy as np

# Load a trained model
model = YOLO('C:/Users/anish/OneDrive/Desktop/FYP/fyp-palm/runs/detect/YOLO_Detection/weights/best.pt')

# Run validation on the test set with plots enabled to generate confusion matrix
metrics = model.val(plots=True)

# Calculate and display YOLO accuracy metrics
print("\n" + "="*60)
print("YOLO MODEL PERFORMANCE METRICS")
print("="*60)

# Box metrics (Detection metrics)
print("\nDetection Metrics:")
print(f"  Precision (P):     {metrics.box.p[0]:.4f}")  # Precision
print(f"  Recall (R):        {metrics.box.r[0]:.4f}")  # Recall
print(f"  F1-Score:          {2 * (metrics.box.p[0] * metrics.box.r[0]) / (metrics.box.p[0] + metrics.box.r[0] + 1e-6):.4f}")

# Mean Average Precision (mAP) - NOT the same as accuracy
print("\nMean Average Precision (mAP):")
print(f"  mAP@0.5:           {metrics.box.map50:.4f}")  # mAP at IoU 0.5
print(f"  mAP@0.5:0.95:      {metrics.box.map:.4f}")    # mAP at IoU 0.5-0.95

# Calculate Accuracy using ACTUAL confusion matrix (TP / (TP + FP + FN))
# This is the same method used in evaluation_metrics.py for consistency
accuracy = None
cm = getattr(metrics, "confusion_matrix", None)

if cm is not None and hasattr(cm, "matrix") and cm.matrix is not None:
    # Get confusion matrix (nc+1, nc+1) - last row/col is background
    cm_matrix = cm.matrix
    
    # Exclude background (last row and column)
    real = cm_matrix[:-1, :-1]
    
    # TP: diagonal elements (correct classifications)
    tp = float(np.trace(real))
    
    # Total predictions
    total_predictions = float(real.sum())
    
    # FP: wrong predictions (total - TP)
    fp = total_predictions - tp
    
    # FN: ground truth objects that were missed (in background column)
    fn = float(cm_matrix[:-1, -1].sum())
    
    # Calculate accuracy
    denom = tp + fp + fn
    accuracy = float(tp / denom) if denom > 0 else 0.0
    
    print("\nAccuracy Calculation (from Confusion Matrix):")
    print(f"  True Positives:    {tp:.0f}")
    print(f"  False Positives:   {fp:.0f}")
    print(f"  False Negatives:   {fn:.0f}")
    print(f"  Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
else:
    # Fallback to estimation method if confusion matrix not available
    precision = metrics.box.p[0]
    recall = metrics.box.r[0]
    
    if precision > 0 and recall > 0:
        TP = 1.0
        FP = TP * (1 - precision) / precision if precision > 0 else 0
        FN = TP * (1 - recall) / recall if recall > 0 else 0
        accuracy = TP / (TP + FP + FN)
    else:
        accuracy = 0.0
    
    print("\nAccuracy Calculation (Estimated from P/R):")
    print(f"  Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("  ⚠️  Note: Using estimated accuracy. Enable plots=True for exact values.")

# Per-class metrics if available
if hasattr(metrics.box, 'ap_class_index'):
    print("\nPer-Class Metrics:")
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        print(f"  Class {class_idx}:")
        print(f"    Precision: {metrics.box.p[i]:.4f}")
        print(f"    Recall:    {metrics.box.r[i]:.4f}")
        print(f"    AP@0.5:    {metrics.box.ap50[i]:.4f}")
        print(f"    AP@0.5:0.95: {metrics.box.ap[i]:.4f}")

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print(f"  Accuracy:          {accuracy * 100:.2f}%")
print(f"  mAP@0.5:           {metrics.box.map50 * 100:.2f}%")
print(f"  mAP@0.5:0.95:      {metrics.box.map * 100:.2f}%")
print("="*60 + "\n")
