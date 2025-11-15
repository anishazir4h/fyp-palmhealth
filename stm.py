
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
import torch
import json
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
import os
from database import PalmDatabase

# Initialize database
db = PalmDatabase()

# Page configuration
st.set_page_config(
    page_title="Palm Health Detection System",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Theme with Readable Fonts
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #FAFAFA !important;
    }
    
    /* Dark theme backgrounds */
    .stApp {
        background-color: #0E1117;
    }
    
    .main {
        background-color: #0E1117;
    }
    
    /* Force all text to be white for readability */
    .stApp, .main, .block-container, p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #3a3a3a;
    }
    
    [data-testid="stSidebar"] * {
        color: #FAFAFA !important;
    }
    
    /* Clean headers */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #FAFAFA !important;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 0.95rem;
        color: #B0B0B0 !important;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Dark metric cards */
    .metric-card {
        background: #262730;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #3a3a3a;
        text-align: center;
        transition: border 0.2s;
    }
    
    .metric-card:hover {
        border-color: #10b981;
    }
    
    .metric-card h3 {
        font-size: 1.75rem;
        margin: 0.5rem 0;
        color: #FAFAFA !important;
    }
    
    .metric-card h4 {
        font-size: 1rem;
        font-weight: 500;
        margin: 0.5rem 0;
        color: #FAFAFA !important;
    }
    
    .metric-card p {
        font-size: 0.875rem;
        color: #B0B0B0 !important;
        margin: 0;
    }
        color: #1a1a1a;
        font-weight: 600;
    }
    
    .metric-card h4 {
        font-size: 1rem;
        font-weight: 500;
        margin: 0.5rem 0;
        color: #1a1a1a;
    }
    
    .metric-card p {
        font-size: 0.875rem;
        color: #666;
        margin: 0;
    }
    
    /* Dark info boxes */
    .info-box {
        background-color: #262730;
        padding: 1.25rem;
        border-radius: 6px;
        border-left: 3px solid #10b981;
        margin: 1rem 0;
        color: #FAFAFA !important;
    }
    
    .warning-box {
        background-color: #262730;
        padding: 1.25rem;
        border-radius: 6px;
        border-left: 3px solid #f59e0b;
        margin: 1rem 0;
        color: #FAFAFA !important;
    }
    
    .error-box {
        background-color: #262730;
        padding: 1.25rem;
        border-radius: 6px;
        border-left: 3px solid #ef4444;
        margin: 1rem 0;
        color: #FAFAFA !important;
    }
    
    /* Clean badges */
    .status-healthy {
        background-color: #10b981;
        color: white;
        padding: 0.375rem 0.875rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        font-size: 0.875rem;
    }
    
    .status-unhealthy {
        background-color: #ef4444;
        color: white;
        padding: 0.375rem 0.875rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        font-size: 0.875rem;
    }
    
    /* Dark palm cards */
    .palm-card {
        background: #262730;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #3a3a3a;
        margin: 0.5rem 0;
        color: #FAFAFA !important;
    }
    
    /* Streamlit metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1a1a1a;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: #666;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Simple buttons */
    .stButton > button {
        background-color: #10b981;
        color: white;
        font-weight: 500;
        border-radius: 6px;
        border: none;
        padding: 0.625rem 1.25rem;
        transition: background 0.2s;
        font-size: 0.875rem;
    }
    
    .stButton > button:hover {
        background-color: #059669;
    }
    
    /* Clean progress bars */
    .stProgress > div > div > div > div {
        background-color: #10b981;
    }
    
    /* Minimal file uploader */
    [data-testid="stFileUploader"] {
        background-color: #262730;
        border-radius: 6px;
        padding: 1.5rem;
        border: 1px dashed #4a4a4a;
        color: #FAFAFA !important;
    }
    
    /* Simple tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #262730;
        border-radius: 6px;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.875rem;
        color: #B0B0B0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0E1117;
        color: #FAFAFA !important;
    }
    
    /* White text for readability */
    p, li, span {
        color: #FAFAFA !important;
        line-height: 1.6;
        font-size: 0.9375rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
        font-weight: 600;
    }
    
    /* Remove extra spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load detection and validation models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try multiple paths for flexibility (local and deployed)
    possible_yolo_paths = [
        "C:/Users/anish/OneDrive/Desktop/FYP/fyp-palm/runs/detect/YOLO_Detection/weights/best.pt",
        "runs/detect/YOLO_Detection/weights/best.pt",
        "models/yolo_best.pt",
        "yolo_best.pt"
    ]
    
    possible_frcnn_paths = [
        "C:/Users/anish/OneDrive/Desktop/FYP/fyp-palm/runs/detect/FasterRCNN_ResNet50_Optimized/weights/best.pt",
        "runs/detect/FasterRCNN_ResNet50_Optimized/weights/best.pt",
        "models/frcnn_best.pt",
        "frcnn_best.pt"
    ]
    
    yolo_model = None
    frcnn_model = None
    yolo_path = None
    frcnn_path = None
    
    # Try to load YOLO model
    for path in possible_yolo_paths:
        try:
            if os.path.exists(path):
                yolo_model = YOLO(path)
                yolo_path = path
                break
        except Exception as e:
            continue
    
    # Try to load Faster R-CNN model
    for path in possible_frcnn_paths:
        try:
            if os.path.exists(path):
                frcnn_model = fasterrcnn_resnet50_fpn(weights=None)
                in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
                frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
                
                frcnn_checkpoint = torch.load(path, map_location=device)
                frcnn_model.load_state_dict(frcnn_checkpoint)
                frcnn_model.to(device)
                frcnn_model.eval()
                frcnn_path = path
                break
        except Exception as e:
            continue
    
    if yolo_model is None or frcnn_model is None:
        missing = []
        if yolo_model is None:
            missing.append("YOLO Detection model")
        if frcnn_model is None:
            missing.append("Faster R-CNN model")
        
        st.error(f"❌ Missing models: {', '.join(missing)}")
        st.info("""
        📌 **For deployment**: Upload model files to the repository:
        - `models/yolo_best.pt` (YOLO model)
        - `models/frcnn_best.pt` (Faster R-CNN model)
        
        Or use Git LFS for large files.
        """)
        return None, None, None
    
    return yolo_model, frcnn_model, f"Detection Model: {yolo_path}\nValidation Model: {frcnn_path}"

def enhance_palm_visibility(image):
    """Enhanced image processing for better palm detection in aerial images"""
    # Convert PIL to OpenCV format
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Increase contrast and brightness
    enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
    
    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 3. Sharpen the image to enhance palm edges
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # 4. Enhance vegetation (palms) - including yellow/unhealthy palms
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    
    # Boost saturation to make yellow palms more visible
    hsv[:,:,1] = cv2.add(hsv[:,:,1], 40)  # Increase saturation globally
    
    # Enhance green vegetation (healthy palms)
    mask_green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    hsv[:,:,1] = cv2.add(hsv[:,:,1], cv2.bitwise_and(np.full_like(hsv[:,:,1], 30), mask_green))
    
    # Enhance yellow vegetation (unhealthy palms) - expanded range to catch yellowing palms
    mask_yellow = cv2.inRange(hsv, (15, 40, 40), (35, 255, 255))  # Yellow hue range
    hsv[:,:,1] = cv2.add(hsv[:,:,1], cv2.bitwise_and(np.full_like(hsv[:,:,1], 50), mask_yellow))
    
    final_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Convert back to PIL format
    return Image.fromarray(cv2.cvtColor(final_enhanced, cv2.COLOR_BGR2RGB))


def make_square_bbox(bbox, image_size):
    """
    Convert rectangular bounding box to square bounding box
    Maintains center point and uses larger dimension
    """
    x1, y1, x2, y2 = bbox
    img_width, img_height = image_size
    
    # Calculate current dimensions
    width = x2 - x1
    height = y2 - y1
    
    # Use the larger dimension for square size
    square_size = max(width, height)
    
    # Calculate center point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate new square coordinates
    half_size = square_size / 2
    new_x1 = center_x - half_size
    new_y1 = center_y - half_size
    new_x2 = center_x + half_size
    new_y2 = center_y + half_size
    
    # Apply boundary constraints
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_width, new_x2)
    new_y2 = min(img_height, new_y2)
    
    # Ensure it's still square after boundary constraints
    actual_width = new_x2 - new_x1
    actual_height = new_y2 - new_y1
    
    if actual_width != actual_height:
        # Use the smaller dimension to ensure it fits in image
        final_size = min(actual_width, actual_height)
        
        # Recenter
        center_x = (new_x1 + new_x2) / 2
        center_y = (new_y1 + new_y2) / 2
        half_final = final_size / 2
        
        new_x1 = max(0, center_x - half_final)
        new_y1 = max(0, center_y - half_final)
        new_x2 = min(img_width, new_x1 + final_size)
        new_y2 = min(img_height, new_y1 + final_size)
    
    return [new_x1, new_y1, new_x2, new_y2]

def multi_scale_detection(model, image, confidence_threshold=0.1):
    """Perform detection at multiple scales to catch palms of different sizes"""
    results_all = []
    
    # Original scale
    results = model.predict(image, conf=confidence_threshold, iou=0.3, max_det=300, verbose=False)
    results_all.extend(results)
    
    # Scale up (zoom in) - helps detect smaller palms
    width, height = image.size
    scaled_up = image.resize((int(width * 1.5), int(height * 1.5)), Image.Resampling.LANCZOS)
    results_scaled = model.predict(scaled_up, conf=confidence_threshold, iou=0.3, max_det=300, verbose=False)
    
    # Scale results back to original coordinates
    if results_scaled and len(results_scaled) > 0 and results_scaled[0].boxes is not None:
        # For simplicity, just use original results for now
        # In a full implementation, you'd scale back coordinates
        pass
    
    return results_all

def detect_with_enhancement(model, image, confidence_threshold=0.1, use_multiscale=True, iou_threshold=0.3):
    """Enhanced detection pipeline combining preprocessing and multi-scale detection"""
    # Step 1: Enhance image for better palm visibility
    enhanced_image = enhance_palm_visibility(image)
    
    # Step 2: Multi-scale detection (optional)
    if use_multiscale:
        results = multi_scale_detection(model, enhanced_image, confidence_threshold)
        return results[0] if results else None
    else:
        # Single scale enhanced detection with aggressive NMS settings for individual palm detection
        results = model.predict(
            enhanced_image, 
            conf=confidence_threshold, 
            iou=0.1,  # VERY low IoU - prevents merging nearby palms
            max_det=2000,  # Allow many individual detections
            agnostic_nms=False,  # Class-specific NMS
            verbose=False
        )
        return results[0] if results else None

def tile_based_detection(model, image, confidence_threshold=0.1, tile_size=256, overlap=0.3):
    """
    Detect palms using sliding window/tiling approach for dense plantations
    OPTIMIZED: Reduced overlap to prevent duplicates
    """
    from PIL import Image
    import numpy as np
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    width, height = image.size
    step = int(tile_size * (1 - overlap))  # 30% overlap - reduced from 80%
    
    all_detections = []
    
    # Moderate overlap to prevent duplicates
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Extract tile
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            # Accept even very small tiles
            if x_end - x < 32 or y_end - y < 32:
                continue
            
            tile = image.crop((x, y, x_end, y_end))
            
            # CRITICAL: Scale up significantly to match training resolution
            # Your model was trained at 416px with large palms
            target_size = 416
            scaled_tile = tile.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Enhance for detection
            enhanced_tile = enhance_palm_visibility(scaled_tile)
            
            # Use model's training characteristics: conf=0.3 works well in validation
            tile_results = model.predict(
                enhanced_tile,
                conf=0.15,    # Lower than validation but not too low
                iou=0.5,      # Higher IoU to prevent duplicates within tiles
                max_det=20,   # Reasonable for small tiles
                verbose=False
            )
            
            # Scale results back to original coordinates
            if tile_results and len(tile_results) > 0 and tile_results[0].boxes is not None:
                boxes = tile_results[0].boxes
                scale_factor = tile_size / target_size  # Scale back down
                
                for i in range(len(boxes)):
                    # Get box coordinates and scale back to tile size
                    box = boxes.xyxy[i].cpu().numpy().copy()
                    box = box * scale_factor  # Scale back to original tile size
                    
                    conf = float(boxes.conf[i])
                    
                    # Size validation for realistic palm sizes
                    box_width = box[2] - box[0]
                    box_height = box[3] - box[1]
                    
                    # Accept palms that make sense for this scale
                    if box_width < 8 or box_height < 8:  # Too small
                        continue
                    if box_width > tile_size * 0.9 or box_height > tile_size * 0.9:  # Too large
                        continue
                    
                    # Convert to square bounding box
                    box = make_square_bbox(box, (tile_size, tile_size))
                    
                    # Adjust coordinates to full image
                    box[0] += x
                    box[1] += y  
                    box[2] += x
                    box[3] += y
                    
                    detection = {
                        'bbox': box,
                        'conf': conf,
                        'cls': int(boxes.cls[i])
                    }
                    all_detections.append(detection)
    
    return all_detections

def merge_overlapping_boxes(detections, overlap_threshold=0.5):
    """
    Remove boxes where TOTAL combined overlap from ALL other boxes >= threshold.
    
    Example: If box A overlaps 30% with box B and 25% with box C,
    total overlap = 55%, so box A is removed if threshold is 50%.
    
    Args:
        detections: List of detections with 'bbox' and 'conf'
        overlap_threshold: Remove box if total overlap >= this (default 0.5 = 50%)
    """
    if not detections or len(detections) <= 1:
        return detections
    
    import torch
    
    # Convert to tensors
    boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d['conf'] for d in detections], dtype=torch.float32)
    
    def calculate_intersection_area(box1, box2):
        """Calculate intersection area between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    def get_box_area(box):
        """Calculate box area"""
        return (box[2] - box[0]) * (box[3] - box[1])
    
    # Sort by confidence (highest first) - keep higher confidence boxes
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    
    for idx in sorted_indices:
        box_i = boxes[idx]
        area_i = get_box_area(box_i)
        
        # Calculate TOTAL overlap from ALL boxes we're already keeping
        total_overlap_area = 0.0
        
        for kept_idx in keep:
            box_j = boxes[kept_idx]
            intersection = calculate_intersection_area(box_i, box_j)
            total_overlap_area += intersection
        
        # Calculate what percentage of current box is covered by other boxes
        overlap_percentage = total_overlap_area / area_i if area_i > 0 else 0.0
        
        # Only keep if total overlap is below threshold
        if overlap_percentage < overlap_threshold:
            keep.append(idx.item())
    
    # Return kept detections in original order
    keep = sorted(keep)
    merged = [detections[i] for i in keep]
    return merged

def custom_nms(boxes, scores, iou_threshold):
    """Custom NMS implementation for when torchvision.ops is unavailable"""
    import torch
    
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by scores
    order = scores.argsort(descending=True)
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i.item())
        
        if len(order) == 1:
            break
        
        # Compute IoU
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU less than threshold
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return torch.tensor(keep, dtype=torch.long)

def merge_tiled_detections(detections, image_size, iou_threshold=0.5):
    """
    Merge overlapping detections from tiled approach
    Higher iou_threshold (0.5) = more aggressive duplicate removal
    """
    if not detections:
        return []
    
    import torch
    
    # Convert to tensors
    boxes = torch.tensor([d['bbox'] for d in detections])
    scores = torch.tensor([d['conf'] for d in detections])
    
    # MINIMAL filtering to preserve individual palm detections
    image_area = image_size[0] * image_size[1]
    valid_indices = []
    
    for i, box in enumerate(boxes):
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        box_ratio = box_area / image_area
        
        # Accept almost all reasonable sizes
        if box_ratio < 0.1 and box_ratio > 0.00001:  # Very wide range
            valid_indices.append(i)
    
    if not valid_indices:
        # Accept everything if no detections pass
        valid_indices = list(range(len(boxes)))
    
    if not valid_indices:
        return detections  # Return all detections as fallback
    
    # Filter to valid detections
    filtered_boxes = boxes[valid_indices]
    filtered_scores = scores[valid_indices]
    
    # Apply NMS with the provided threshold to remove duplicates
    # Use custom NMS implementation for Streamlit Cloud compatibility
    try:
        from torchvision.ops import nms
        keep_indices = nms(filtered_boxes, filtered_scores, iou_threshold)
    except:
        # Fallback to custom NMS if torchvision.ops fails
        keep_indices = custom_nms(filtered_boxes, filtered_scores, iou_threshold)
    
    # Return kept detections
    final_indices = [valid_indices[i] for i in keep_indices]
    merged = [detections[i] for i in final_indices]
    
    return merged

def auto_detect_palms(model, image, confidence_threshold=0.05, validation_model=None):
    """
    SIMPLE 2-STEP: YOLO detection → Faster R-CNN validation
    - Step 1: YOLO detects all palms (fast, single pass)
    - Step 2: Faster R-CNN validates to confirm individual tree crowns (automatic, transparent)
    """
    
    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: YOLO Detection (single pass, no tiling)
        status_text.text("🔍 Step 1/2: Detecting palms with YOLO...")
        progress_bar.progress(10)
        
        results = model.predict(
            image,
            conf=confidence_threshold,
            iou=0.45,
            max_det=300,
            verbose=False
        )
    except Exception as e:
        st.error(f"Error during YOLO detection: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []
    
    try:
        progress_bar.progress(30)

        # Quick sanity
        if not results or len(results) == 0 or results[0].boxes is None:
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            st.warning("⚠️ No palms detected. Try lowering the confidence threshold.")
            return results
    except Exception as e:
        st.error(f"Error checking YOLO results: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []

    # If YOLO produced very few detections but image looks dense, fall back to tile-based detection
    initial_count = len(results[0].boxes) if results[0].boxes is not None else 0

    # Heuristic: compute green density to detect dense plantations
    status_text.text("🌿 Analyzing image density...")
    progress_bar.progress(40)
    
    try:
        img_arr = np.array(image.convert('RGB'))
        hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
        green_density = mask.sum() / (255.0 * img_arr.shape[0] * img_arr.shape[1])
    except Exception:
        green_density = 0.0

    use_tiling = False
    # If image is large and green density is high or YOLO found very few boxes, switch to tile-based detection
    if (green_density > 0.12 and (image.size[0] * image.size[1] > 512 * 512)) or initial_count < 5:
        use_tiling = True

    # If tiling is selected, run tile-based detection then merge
    if use_tiling:
        try:
            status_text.text("🔲 Using tile-based detection for dense plantation...")
            progress_bar.progress(50)
            
            tiled = tile_based_detection(model, image, confidence_threshold=max(0.12, confidence_threshold), tile_size=416, overlap=0.25)
            merged = merge_tiled_detections(tiled, image.size, iou_threshold=0.35)

            # Convert merged to results-like structure
            detections = merged
        except Exception as e:
            st.error(f"Error during tiled detection: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return []
    else:
        try:
            # Convert YOLO results to detection list
            status_text.text("📋 Processing YOLO detections...")
            progress_bar.progress(50)
            
            boxes = results[0].boxes
            detections = []
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                detections.append({
                    'bbox': bbox,
                    'conf': float(boxes.conf[i]),
                    'cls': int(boxes.cls[i])
                })
        except Exception as e:
            st.error(f"Error processing YOLO boxes: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return []
    
    progress_bar.progress(60)

    # If a validation model is provided, validate detections
    if validation_model is not None and detections:
        try:
            status_text.text("✓ Step 2/2: Validating detections...")
            progress_bar.progress(70)
            
            validated = validate_with_faster_rcnn_2batch(detections, image, validation_model)
            detections = validated
            
            progress_bar.progress(85)
        except Exception as e:
            st.error(f"Error during validation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            # Continue with unvalidated detections
            progress_bar.progress(85)
    
    # MERGE OVERLAPPING BOXES - Remove boxes where 50%+ of area is covered by other boxes combined
    if detections and len(detections) > 1:
        try:
            status_text.text("🔄 Removing duplicate detections...")
            progress_bar.progress(90)
            
            before_merge = len(detections)
            detections = merge_overlapping_boxes(detections, overlap_threshold=0.5)
            after_merge = len(detections)
        except Exception as e:
            st.error(f"Error merging overlapping boxes: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            # Continue with unmerged detections
            progress_bar.progress(90)
    
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    # Convert back to YOLO-like result
    class MockBoxes:
        def __init__(self, detections):
            if len(detections) == 0:
                self.xyxy = torch.empty((0,4))
                self.conf = torch.empty((0,))
                self.cls = torch.empty((0,), dtype=torch.int64)
            else:
                self.xyxy = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
                self.conf = torch.tensor([d['conf'] for d in detections], dtype=torch.float32)
                self.cls = torch.tensor([d['cls'] for d in detections], dtype=torch.int64)

        def __len__(self):
            return len(self.xyxy)

    class MockResult:
        def __init__(self, detections):
            self.boxes = MockBoxes(detections) if detections else None

    final_results = [MockResult(detections)]
    return final_results


def validate_with_faster_rcnn_2batch(detections, image, faster_rcnn_model, conf_threshold=0.10):
    """
    ACCEPT ALL INITIAL DETECTIONS - Use validation model only for classification, not rejection
    - Filters out elongated rectangles (aspect ratio > 1.5)
    - Uses validation model to reclassify health status only
    - Does NOT reject any detections based on validation model confidence
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validated_detections = []
    
    # Filter out invalid elongated rectangles only
    valid_detections = []
    
    for detection in detections:
        # Filter out invalid elongated rectangles (too long horizontally or vertically)
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Calculate aspect ratio (should be roughly square for palm trees)
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            # Reject boxes that are too elongated (aspect ratio > 1.5)
            # Palm trees should be roughly circular/square from aerial view
            if aspect_ratio > 1.5:  # VERY STRICT: 1.5 to filter rectangles
                continue  # Skip this detection
        
        valid_detections.append(detection)
    
    # If no Faster R-CNN model, return all valid detections as-is
    if faster_rcnn_model is None:
        return valid_detections
    
    # Process all valid detections with Faster R-CNN for classification
    # Split into batches for memory efficiency
    batch_size = 32
    all_classified = []
    
    for batch_start in range(0, len(valid_detections), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_detections))
        batch = valid_detections[batch_start:batch_end]
        
        batch_tensors = []
        batch_items = []
        
        for detection in batch:
            bbox = detection['bbox']
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(image.width, int(bbox[2]))
            y2 = min(image.height, int(bbox[3]))
            
            if x2 - x1 < 5 or y2 - y1 < 5:
                # Keep detection with original classification if crop too small
                all_classified.append(detection)
                continue
            
            crop = image.crop((x1, y1, x2, y2))
            target_size = 224
            crop_resized = crop.resize((target_size, target_size), Image.BILINEAR)
            crop_tensor = TF.to_tensor(crop_resized).to(device)
            
            batch_tensors.append(crop_tensor)
            batch_items.append(detection)
        
        # Run Faster R-CNN on batch
        if len(batch_tensors) > 0:
            with torch.no_grad():
                frcnn_preds = faster_rcnn_model(batch_tensors)
            
            # Update classifications based on Faster R-CNN
            for detection, frcnn_pred in zip(batch_items, frcnn_preds):
                if 'boxes' in frcnn_pred and len(frcnn_pred['boxes']) > 0:
                    scores = frcnn_pred['scores'].cpu().numpy()
                    labels = frcnn_pred['labels'].cpu().numpy() if 'labels' in frcnn_pred else None
                    
                    # Get top prediction
                    top_idx = int(scores.argmax())
                    top_score = float(scores[top_idx])
                    
                    # Update classification from Faster R-CNN
                    if labels is not None:
                        detection['cls'] = int(labels[top_idx] - 1)  # Convert to 0/1 (Unhealthy/Healthy)
                    detection['conf'] = max(detection['conf'], top_score)  # Use higher confidence
                
                # ALWAYS keep the detection regardless of Faster R-CNN result
                all_classified.append(detection)
    
    return all_classified

def get_confidence_color(confidence):
    """Get color class based on confidence score"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def format_confidence(confidence):
    """Format confidence score with appropriate styling"""
    color_class = get_confidence_color(confidence)
    return f'<span class="{color_class}">{confidence:.1%}</span>'

def create_detection_summary(results):
    """Create a detailed summary of detection results"""
    if not results or len(results) == 0:
        return None
    
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return {
            'total_palms': 0,
            'healthy_palms': 0,
            'unhealthy_palms': 0,
            'avg_confidence': 0,
            'detections': []
        }
    
    boxes = result.boxes
    class_names = ['PalmAnom', 'PalmSan']  # 0: Anomalous, 1: Healthy
    
    detections = []
    healthy_count = 0
    unhealthy_count = 0
    total_confidence = 0
    
    for i in range(len(boxes)):
        cls = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        
        # Handle different box formats (YOLO vs MockBoxes)
        if hasattr(boxes.xyxy[i], 'cpu'):
            box = boxes.xyxy[i].cpu().numpy()
        else:
            box = boxes.xyxy[i].numpy() if hasattr(boxes.xyxy[i], 'numpy') else boxes.xyxy[i]
        
        # Determine health status
        is_healthy = (cls == 1)  # PalmSan = Healthy
        if is_healthy:
            healthy_count += 1
            status = "Healthy"
            status_color = "#28a745"
        else:
            unhealthy_count += 1
            status = "Unhealthy"
            status_color = "#dc3545"
        
        detections.append({
            'id': i + 1,
            'status': status,
            'status_color': status_color,
            'confidence': conf,
            'class_name': class_names[cls],
            'bbox': box
        })
        
        total_confidence += conf
    
    avg_confidence = total_confidence / len(boxes) if len(boxes) > 0 else 0
    
    return {
        'total_palms': len(boxes),
        'healthy_palms': healthy_count,
        'unhealthy_palms': unhealthy_count,
        'avg_confidence': avg_confidence,
        'detections': detections
    }

def crop_individual_palms(image, results):
    """
    Crop individual palm trees from the image based on detection results
    Returns a list of cropped images with metadata
    """
    if not results or len(results) == 0:
        return []
    
    result = results[0]
    if not hasattr(result, 'boxes') or result.boxes is None:
        return []
    
    # Convert PIL Image to numpy array if needed
    if hasattr(image, 'size'):  # PIL Image
        import numpy as np
        img_array = np.array(image)
        img_width, img_height = image.size
    else:  # Already numpy array
        img_array = image
        img_height, img_width = image.shape[:2]
    
    cropped_palms = []
    class_names = ['PalmAnom', 'PalmSan']  # 0: Anomalous, 1: Healthy
    boxes = result.boxes
    
    # Handle both MockBoxes and regular YOLO boxes
    if hasattr(boxes, '__len__'):
        num_boxes = len(boxes)
    else:
        # For MockBoxes, check if it has xyxy attribute and get length from there
        if hasattr(boxes, 'xyxy') and hasattr(boxes.xyxy, '__len__'):
            num_boxes = len(boxes.xyxy)
        else:
            return []
    
    for i in range(num_boxes):
        # Get bounding box coordinates - handle different box formats
        if hasattr(boxes, 'xyxy'):
            if hasattr(boxes.xyxy[i], 'cpu'):
                bbox = boxes.xyxy[i].cpu().numpy()
            elif hasattr(boxes.xyxy[i], 'numpy'):
                bbox = boxes.xyxy[i].numpy()
            else:
                bbox = boxes.xyxy[i]
        else:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Since bounding boxes are already square, just add minimal padding
        padding = 5  # Small padding in pixels
        
        # Apply padding with image boundary checks
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(img_width, x2 + padding)
        y2_padded = min(img_height, y2 + padding)
        
        # Skip if the cropped area is too small
        if x2_padded <= x1_padded or y2_padded <= y1_padded:
            continue
            
        # Crop the palm from the numpy array (bounding box is already square)
        cropped_palm = img_array[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Get metadata - handle different confidence formats
        try:
            if hasattr(boxes, 'conf'):
                if hasattr(boxes.conf[i], 'cpu'):
                    conf = float(boxes.conf[i].cpu().numpy())
                elif hasattr(boxes.conf[i], 'numpy'):
                    conf = float(boxes.conf[i].numpy())
                else:
                    conf = float(boxes.conf[i])
            else:
                conf = 0.5  # Default confidence
        except:
            conf = 0.5
            
        # Get class - handle different class formats
        try:
            if hasattr(boxes, 'cls'):
                if hasattr(boxes.cls[i], 'cpu'):
                    cls = int(boxes.cls[i].cpu().numpy())
                elif hasattr(boxes.cls[i], 'numpy'):
                    cls = int(boxes.cls[i].numpy())
                else:
                    cls = int(boxes.cls[i])
            else:
                cls = 0  # Default class
        except:
            cls = 0
            
        is_healthy = (cls == 1)  # PalmSan = Healthy
        status = "Healthy" if is_healthy else "Unhealthy"
        class_name = class_names[cls] if cls < len(class_names) else "Unknown"
        status_color = "#28a745" if is_healthy else "#dc3545"
        
        cropped_palms.append({
            'id': i + 1,
            'image': cropped_palm,
            'confidence': conf,
            'class': cls,
            'class_name': class_name,
            'status': status,
            'status_color': status_color,
            'bbox': bbox,
            'cropped_bbox': (x1_padded, y1_padded, x2_padded, y2_padded)
        })
    
    return cropped_palms

def create_health_chart(summary):
    """Create a pie chart showing palm health distribution"""
    if summary['total_palms'] == 0:
        return None
    
    labels = ['Healthy Palms', 'Unhealthy Palms']
    values = [summary['healthy_palms'], summary['unhealthy_palms']]
    colors = ['#28a745', '#dc3545']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent+value',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="Palm Health Distribution",
        title_x=0.5,
        font=dict(size=12),
        height=400,
        showlegend=True
    )
    
    return fig

def main():
    # Sidebar Navigation
    st.sidebar.title("🌴 Palm Health System")
    st.sidebar.markdown("---")
    
    # Initialize current page in session state if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"
    
    # Check if navigation requested from button click
    if 'navigate_to' in st.session_state:
        st.session_state.current_page = st.session_state.navigate_to
        del st.session_state.navigate_to
    
    # Navigation radio - synced with current page
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload & Analyze", "📊 Dashboard", "📋 Detection History"],
        index=["🏠 Home", "📤 Upload & Analyze", "📊 Dashboard", "📋 Detection History"].index(st.session_state.current_page),
        label_visibility="collapsed",
        key="main_navigation"
    )
    
    # Update current page if radio changed
    if page != st.session_state.current_page:
        st.session_state.current_page = page
    
    # Statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    stats = db.get_statistics()
    st.sidebar.metric("Total Images", stats['total_images'])
    st.sidebar.metric("Total Palms", stats['total_palms'])
    st.sidebar.metric("Health Rate", f"{stats['avg_health_rate']:.1f}%")
    
    # Route to pages based on current page state
    current = st.session_state.current_page
    if current == "🏠 Home":
        show_home_page()
    elif current == "📤 Upload & Analyze":
        show_upload_page()
    elif current == "📊 Dashboard":
        show_dashboard_page()
    elif current == "📋 Detection History":
        show_history_page()

def show_home_page():
    """Home/Landing page"""
    st.markdown('<h1 class="main-header">🌴 Palm Health Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Aerial Palm Plantation Health Monitoring</p>', unsafe_allow_html=True)
    
    # Hero CTA - Primary action
    st.markdown("""
    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                padding: 3rem 2rem; 
                border-radius: 12px; 
                text-align: center; 
                margin: 2rem 0 3rem 0;">
        <h2 style="color: white; font-size: 1.75rem; margin: 0 0 1rem 0; font-weight: 600;">
            👋 Get Started in 3 Simple Steps
        </h2>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.1rem; margin: 0 0 2rem 0;">
            Upload your aerial palm images → AI analyzes health → View instant results
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Primary CTA button
    col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
    with col_cta2:
        if st.button("🚀 Start Analyzing Palm Images", key="primary_cta", use_container_width=True, type="primary"):
            st.session_state.navigate_to = "📤 Upload & Analyze"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick stats overview
    stats = db.get_statistics()
    if stats['total_images'] > 0:
        st.markdown("### 📊 Your Statistics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Images Analyzed", stats['total_images'])
        with metric_col2:
            st.metric("Total Palms", stats['total_palms'])
        with metric_col3:
            st.metric("Health Rate", f"{stats['avg_health_rate']:.1f}%")
        with metric_col4:
            st.metric("Unhealthy Palms", stats['total_unhealthy'])
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Secondary navigation options
    st.markdown("### 🎯 What would you like to do?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #262730; padding: 1.5rem; border-radius: 8px; border: 1px solid #3a3a3a; text-align: center; min-height: 160px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📤</div>
            <h4 style="margin: 0.5rem 0; color: #FAFAFA;">Upload & Analyze</h4>
            <p style="font-size: 0.875rem; color: #B0B0B0; margin: 0.5rem 0;">Upload images for instant AI detection</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Upload", key="nav_upload_btn", use_container_width=True):
            st.session_state.navigate_to = "📤 Upload & Analyze"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: #262730; padding: 1.5rem; border-radius: 8px; border: 1px solid #3a3a3a; text-align: center; min-height: 160px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📊</div>
            <h4 style="margin: 0.5rem 0; color: #FAFAFA;">View Dashboard</h4>
            <p style="font-size: 0.875rem; color: #B0B0B0; margin: 0.5rem 0;">Analytics, charts & insights</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Dashboard", key="nav_dashboard_btn", use_container_width=True):
            st.session_state.navigate_to = "📊 Dashboard"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style="background: #262730; padding: 1.5rem; border-radius: 8px; border: 1px solid #3a3a3a; text-align: center; min-height: 160px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">�</div>
            <h4 style="margin: 0.5rem 0; color: #FAFAFA;">Browse History</h4>
            <p style="font-size: 0.875rem; color: #B0B0B0; margin: 0.5rem 0;">View all past detections</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to History", key="nav_history_btn", use_container_width=True):
            st.session_state.navigate_to = "📋 Detection History"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recent activity (if any)
    recent = db.get_recent_detections(limit=3)
    if recent:
        st.markdown("### 🕒 Recent Activity")
        for detection in recent:
            det_id, timestamp, img_name, total, healthy, unhealthy, health_rate, img_path = detection
            
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{img_name}** - {timestamp}")
                st.caption(f"{total} palms detected • {health_rate:.1f}% healthy")
            with col_b:
                if health_rate >= 80:
                    st.markdown('<span class="status-healthy">Healthy</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-unhealthy">Needs Attention</span>', unsafe_allow_html=True)
            st.markdown("<hr style='margin: 0.5rem 0; border-color: #e0e0e0;'>", unsafe_allow_html=True)
    else:
        # First time user guidance
        st.markdown("""
        <div class="info-box" style="margin-top: 2rem;">
            <h4>👋 Welcome! Here's how to start:</h4>
            <ol style="margin: 0.5rem 0; padding-left: 1.5rem; line-height: 1.8;">
                <li>Click the <strong>"🚀 Start Analyzing Palm Images"</strong> button above</li>
                <li>Upload your aerial palm plantation images (JPG, PNG)</li>
                <li>AI will automatically detect and classify palm health</li>
                <li>View results instantly and explore analytics</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # System info at bottom
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ About the System"):
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            **AI Model:**
            - Architecture: YOLOv8n
            - Accuracy: 97%
            - Classes: Healthy, Unhealthy
            """)
        with col_info2:
            st.markdown("""
            **Features:**
            - Real-time detection
            - Auto-save to database
            - Comprehensive analytics
            """)

def show_upload_page():
    """Upload and analyze page"""
    st.markdown('<h1 class="main-header">📤 Upload & Analyze Palm Images</h1>', unsafe_allow_html=True)
    
    # Load models
    yolo_model, frcnn_model, model_info = load_model()
    
    if yolo_model is None:
        st.error("❌ Failed to load models")
        return
    
    # Upload section
    uploaded_files = st.file_uploader(
        "Upload aerial palm images for health analysis",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Supported formats: JPG, JPEG, PNG",
        key="upload_page_uploader"
    )
    
    if uploaded_files:
        st.markdown(f"### 🔍 Analysis Results ({len(uploaded_files)} image(s))")
        
        # Process each image
        for idx, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"🌴 {uploaded_file.name}", expanded=(idx == 0)):
                # Load image
                img = Image.open(uploaded_file)
                
                # Save image temporarily
                img_dir = "uploaded_images"
                os.makedirs(img_dir, exist_ok=True)
                img_path = os.path.join(img_dir, uploaded_file.name)
                img.save(img_path)
                
                # Detect with YOLO + Faster R-CNN validation (automatic)
                with st.spinner("🤖 Analyzing..."):
                    results = auto_detect_palms(yolo_model, img, confidence_threshold=0.05, validation_model=frcnn_model)
                
                # Create summary
                summary = create_detection_summary(results)
                
                if summary and summary['total_palms'] > 0:
                    # Save to database
                    detection_id = db.save_detection(
                        uploaded_file.name,
                        img_path,
                        summary,
                        img.size
                    )
                    
                    st.success(f"✅ Detection saved to database (ID: {detection_id})")
                    
                    # Display results
                    col_img, col_stats = st.columns([2, 1])
                    
                    with col_img:
                        st.markdown("#### 🖼️ Detection Visualization")
                        # Draw bounding boxes on image
                        if results and len(results) > 0 and results[0].boxes is not None:
                            try:
                                # Try to use YOLO plot if available
                                if hasattr(results[0], 'plot'):
                                    res_plotted = results[0].plot()
                                    st.image(res_plotted, use_container_width=True)
                                else:
                                    # Manual drawing for MockResult
                                    import cv2
                                    img_array = np.array(img)
                                    img_draw = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                                    
                                    boxes = results[0].boxes
                                    for i in range(len(boxes)):
                                        # Get box coordinates
                                        if hasattr(boxes.xyxy[i], 'cpu'):
                                            box = boxes.xyxy[i].cpu().numpy()
                                        else:
                                            box = boxes.xyxy[i]
                                        
                                        x1, y1, x2, y2 = map(int, box)
                                        
                                        # Get class and confidence
                                        cls = int(boxes.cls[i]) if hasattr(boxes.cls[i], 'cpu') else int(boxes.cls[i])
                                        conf = float(boxes.conf[i]) if hasattr(boxes.conf[i], 'cpu') else float(boxes.conf[i])
                                        
                                        # Color based on class (green for healthy, red for unhealthy)
                                        color = (0, 255, 0) if cls == 1 else (0, 0, 255)
                                        
                                        # Draw rectangle
                                        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
                                        
                                        # Add label
                                        label = f"{'Healthy' if cls == 1 else 'Unhealthy'} {conf:.2f}"
                                        cv2.putText(img_draw, label, (x1, y1 - 10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                    
                                    # Convert back to RGB for display
                                    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                                    st.image(img_draw, use_container_width=True)
                            except Exception as e:
                                st.image(img, use_container_width=True)
                                st.caption(f"Note: Showing original image (visualization error)")
                        else:
                            st.image(img, use_container_width=True)
                    
                    with col_stats:
                        st.markdown("#### 📊 Detection Summary")
                        st.metric("Total Palms", summary['total_palms'])
                        st.metric("Healthy", summary['healthy_palms'])
                        st.metric("Unhealthy", summary['unhealthy_palms'])
                        health_rate = (summary['healthy_palms'] / summary['total_palms'] * 100)
                        st.metric("Health Rate", f"{health_rate:.1f}%")
                        st.progress(health_rate / 100)
                    
                    # Individual Palm Crops Section
                    st.markdown("---")
                    st.markdown("#### 🌴 Individual Palm Tree Analysis")
                    
                    # Crop individual palms
                    cropped_palms = crop_individual_palms(img, results)
                    
                    if cropped_palms:
                        # Display cropped palms in a grid
                        cols_per_row = 3
                        rows = (len(cropped_palms) + cols_per_row - 1) // cols_per_row
                        
                        for row in range(rows):
                            cols = st.columns(cols_per_row)
                            for col_idx in range(cols_per_row):
                                palm_idx = row * cols_per_row + col_idx
                                if palm_idx < len(cropped_palms):
                                    palm = cropped_palms[palm_idx]
                                    
                                    with cols[col_idx]:
                                        # Display cropped palm image
                                        st.image(
                                            palm['image'],
                                            caption=f"Palm {palm['id']}: {palm['status']}",
                                            use_container_width=True
                                        )
                                        
                                        # Status badge with color
                                        status_emoji = "✅" if palm['status'] == "Healthy" else "⚠️"
                                        confidence_color = "🟢" if palm['confidence'] > 0.7 else "🟡" if palm['confidence'] > 0.4 else "🔴"
                                        
                                        st.markdown(f"""
                                        <div style="text-align: center; padding: 5px; background-color: {palm['status_color']}20; border-radius: 5px; margin: 5px 0;">
                                            <strong>{status_emoji} {palm['status']}</strong><br>
                                            {confidence_color} {palm['confidence']:.1%} confidence
                                        </div>
                                        """, unsafe_allow_html=True)
                else:
                    st.warning("No palms detected")

def show_dashboard_page():
    """Analytics dashboard page - similar to Power BI"""
    st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    stats = db.get_statistics()
    
    if stats['total_images'] == 0:
        st.info("📈 No data yet. Upload images to see analytics!")
        return
    
    # KPI Cards
    st.markdown("### 📈 Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">{stats['total_images']}</h3>
            <p style="margin: 0; color: #666;">Total Images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #2E8B57; margin: 0;">{stats['total_palms']}</h3>
            <p style="margin: 0; color: #666;">Total Palms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #4CAF50; margin: 0;">{stats['total_healthy']}</h3>
            <p style="margin: 0; color: #666;">Healthy Palms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #F44336; margin: 0;">{stats['total_unhealthy']}</h3>
            <p style="margin: 0; color: #666;">Unhealthy Palms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FF9800; margin: 0;">{stats['avg_health_rate']:.1f}%</h3>
            <p style="margin: 0; color: #666;">Avg Health Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row 1
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Pie chart - Health Distribution
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Healthy', 'Unhealthy'],
            values=[stats['total_healthy'], stats['total_unhealthy']],
            hole=0.4,
            marker_colors=['#4CAF50', '#F44336'],
            textinfo='label+percent+value'
        )])
        fig_pie.update_layout(
            title="Overall Health Distribution",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with chart_col2:
        # Bar chart - Comparison
        fig_bar = go.Figure(data=[
            go.Bar(name='Healthy', x=['Palms'], y=[stats['total_healthy']], marker_color='#4CAF50'),
            go.Bar(name='Unhealthy', x=['Palms'], y=[stats['total_unhealthy']], marker_color='#F44336')
        ])
        fig_bar.update_layout(
            title="Health Comparison",
            height=400,
            barmode='group'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Charts row 2
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        # Trend chart
        if stats['trend_data']:
            trend_df = pd.DataFrame(stats['trend_data'], columns=['Timestamp', 'Health Rate', 'Total Palms'])
            trend_df = trend_df.iloc[::-1]  # Reverse to show oldest first
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_df['Timestamp'],
                y=trend_df['Health Rate'],
                mode='lines+markers',
                name='Health Rate',
                line=dict(color='#2E8B57', width=3),
                marker=dict(size=10)
            ))
            fig_trend.update_layout(
                title="Health Rate Trend (Last 10 Detections)",
                xaxis_title="Detection Time",
                yaxis_title="Health Rate (%)",
                height=400
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with chart_col4:
        # Gauge chart for average health
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=stats['avg_health_rate'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Health Rate"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#2E8B57"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 80], 'color': "#ffffcc"},
                    {'range': [80, 100], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Individual Palm Analysis Section
    st.markdown("---")
    st.markdown("### 🌴 All Detected Palm Trees")
    
    # Get ALL individual palms from all detections
    all_palms = db.get_all_individual_palms()
    
    if all_palms:
        # Count statistics
        total_all_palms = len(all_palms)
        healthy_all = sum(1 for p in all_palms if p[5] == "Healthy")
        unhealthy_all = total_all_palms - healthy_all
        avg_conf_all = sum(p[6] for p in all_palms) / total_all_palms if total_all_palms > 0 else 0
        
        st.markdown(f"**Total Trees in Database:** {total_all_palms} | ✅ {healthy_all} Healthy | ⚠️ {unhealthy_all} Unhealthy | 📊 {(healthy_all/total_all_palms*100):.1f}% Health Rate")
        
        # SECTION 1: Cropped Images Grid (Latest Detection Only)
        st.markdown("#### 📸 Latest Detection - Individual Trees")
        
        recent_with_palms = db.get_recent_detections(limit=1)
        if recent_with_palms:
            det_id, timestamp, img_name, total, healthy, unhealthy, health_rate, img_path = recent_with_palms[0]
            st.markdown(f"**Image:** {img_name} - {timestamp}")
            
            individual_palms = db.get_individual_palms(det_id)
            
            if individual_palms and img_path and os.path.exists(img_path):
                orig_img = Image.open(img_path)
                img_array = np.array(orig_img)
                
                # Create grid layout for palm images
                cols_per_row = 6
                rows = (len(individual_palms) + cols_per_row - 1) // cols_per_row
                
                for row in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        palm_idx = row * cols_per_row + col_idx
                        if palm_idx < len(individual_palms):
                            palm_number, status, confidence, bbox_json = individual_palms[palm_idx]
                            bbox = json.loads(bbox_json)
                            
                            with cols[col_idx]:
                                # Crop palm from image
                                x1, y1, x2, y2 = map(int, bbox)
                                padding = 5
                                x1_padded = max(0, x1 - padding)
                                y1_padded = max(0, y1 - padding)
                                x2_padded = min(img_array.shape[1], x2 + padding)
                                y2_padded = min(img_array.shape[0], y2 + padding)
                                
                                # Ensure valid crop dimensions
                                if x2_padded > x1_padded and y2_padded > y1_padded:
                                    cropped_palm = img_array[y1_padded:y2_padded, x1_padded:x2_padded]
                                    
                                    # Verify cropped image is not empty
                                    if cropped_palm.size > 0:
                                        # Display cropped image (small size)
                                        st.image(cropped_palm, use_container_width=True)
                                    else:
                                        st.warning(f"Invalid crop for tree #{palm_number}")
                                else:
                                    st.warning(f"Invalid bbox for tree #{palm_number}")
                                
                                # Status info
                                status_emoji = "✅" if status == "Healthy" else "⚠️"
                                status_color = "#28a745" if status == "Healthy" else "#dc3545"
                                
                                st.markdown(f"""
                                <div style="text-align: center; padding: 5px; background-color: {status_color}20; border-radius: 5px; margin-top: 5px;">
                                    <strong style="font-size: 14px;">#{palm_number}</strong><br>
                                    <span style="font-size: 13px;">{status_emoji} {status}</span><br>
                                    <span style="font-size: 13px;">{confidence:.0%}</span>
                                </div>
                                """, unsafe_allow_html=True)
        
        # SECTION 2: Complete Table of ALL Trees from ALL Images
        st.markdown("---")
        st.markdown("#### 📋 Complete Detection Table - All Trees from All Images")
        
        # Create table data for ALL palms
        table_data = []
        tree_counter = 1
        for detection_id, image_name, timestamp, image_path, palm_number, status, confidence, bbox_json in all_palms:
            status_emoji = "✅" if status == "Healthy" else "⚠️"
            
            table_data.append({
                'Global ID': tree_counter,
                'Image': image_name[:30] + "..." if len(image_name) > 30 else image_name,
                'Tree #': palm_number,
                'Status': f"{status_emoji} {status}",
                'Confidence': f"{confidence:.1%}",
                'Detection Date': timestamp[:16]
            })
            tree_counter += 1
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Apply custom CSS for better readability
        st.markdown("""
        <style>
        .dataframe {
            font-size: 15px !important;
            color: black !important;
        }
        .dataframe th {
            font-size: 16px !important;
            font-weight: bold !important;
            background-color: #2E8B57 !important;
            color: white !important;
        }
        .dataframe td {
            font-size: 15px !important;
            padding: 10px !important;
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Style the dataframe
        def highlight_status(row):
            if '✅' in row['Status']:
                return ['background-color: #d4edda; font-size: 15px; color: black;'] * len(row)
            else:
                return ['background-color: #f8d7da; font-size: 15px; color: black;'] * len(row)
        
        styled_df = df.style.apply(highlight_status, axis=1).set_properties(**{
            'font-size': '15px',
            'text-align': 'left',
            'color': 'black'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=500)
        
        # Summary statistics below table
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Trees", total_all_palms)
        with col_stat2:
            st.metric("Healthy", healthy_all, delta=f"{(healthy_all/total_all_palms*100):.1f}%")
        with col_stat3:
            st.metric("Unhealthy", unhealthy_all, delta=f"{(unhealthy_all/total_all_palms*100):.1f}%", delta_color="inverse")
        with col_stat4:
            st.metric("Avg Confidence", f"{avg_conf_all:.1%}")
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Actionable Insights")
    
    if stats['avg_health_rate'] < 50:
        st.markdown("""
        <div class="error-box">
            <h4>⚠️ Critical Health Alert</h4>
            <p>Overall plantation health is below 50%. Immediate action recommended:</p>
            <ul>
                <li>Conduct detailed inspection of unhealthy palms</li>
                <li>Check for pest infestations or diseases</li>
                <li>Review irrigation and fertilization practices</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif stats['avg_health_rate'] < 80:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Moderate Health Concern</h4>
            <p>Health rate is moderate. Consider:</p>
            <ul>
                <li>Monitor unhealthy palms closely</li>
                <li>Implement preventive maintenance</li>
                <li>Schedule regular inspections</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>✅ Good Plantation Health</h4>
            <p>Overall health is good. Maintain current practices:</p>
            <ul>
                <li>Continue regular monitoring</li>
                <li>Keep up with maintenance schedule</li>
                <li>Document best practices</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_history_page():
    """Detection history page"""
    st.markdown('<h1 class="main-header">📋 Detection History</h1>', unsafe_allow_html=True)
    
    # Get all detections
    all_detections = db.get_all_detections()
    
    if not all_detections:
        st.info("No detection history yet. Upload images to get started!")
        return
    
    # Filters
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        filter_health = st.selectbox(
            "Filter by Health",
            ["All", "Healthy (>80%)", "Moderate (50-80%)", "Unhealthy (<50%)"]
        )
    
    with col_filter2:
        sort_by = st.selectbox(
            "Sort by",
            ["Most Recent", "Oldest First", "Most Palms", "Least Palms"]
        )
    
    with col_filter3:
        results_per_page = st.selectbox(
            "Results per page",
            [10, 25, 50, 100]
        )
    
    # Apply filters and sorting
    filtered_data = []
    for det in all_detections:
        det_id, timestamp, img_name, total, healthy, unhealthy, health_rate, avg_conf, img_path = det
        
        # Apply health filter
        if filter_health == "Healthy (>80%)" and health_rate <= 80:
            continue
        elif filter_health == "Moderate (50-80%)" and (health_rate < 50 or health_rate > 80):
            continue
        elif filter_health == "Unhealthy (<50%)" and health_rate >= 50:
            continue
        
        filtered_data.append(det)
    
    # Apply sorting
    if sort_by == "Oldest First":
        filtered_data = filtered_data[::-1]
    elif sort_by == "Most Palms":
        filtered_data = sorted(filtered_data, key=lambda x: x[3], reverse=True)
    elif sort_by == "Least Palms":
        filtered_data = sorted(filtered_data, key=lambda x: x[3])
    
    # Display results
    st.markdown(f"### Showing {len(filtered_data)} results")
    
    # Create dataframe for table view
    df_data = []
    for det in filtered_data[:results_per_page]:
        det_id, timestamp, img_name, total, healthy, unhealthy, health_rate, avg_conf, img_path = det
        df_data.append({
            'ID': det_id,
            'Timestamp': timestamp,
            'Image': img_name,
            'Total Palms': total,
            'Healthy': healthy,
            'Unhealthy': unhealthy,
            'Health Rate (%)': f"{health_rate:.1f}",
            'Avg Confidence': f"{avg_conf:.1%}"
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Detailed view
    st.markdown("---")
    st.markdown("### 🔍 Detailed View")
    
    for det in filtered_data[:results_per_page]:
        det_id, timestamp, img_name, total, healthy, unhealthy, health_rate, avg_conf, img_path = det
        
        with st.expander(f"📸 {img_name} - {timestamp}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if img_path and os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                else:
                    st.info("Image not found")
            
            with col2:
                st.markdown("#### 📊 Statistics")
                st.metric("Total Palms", total)
                st.metric("Healthy", healthy)
                st.metric("Unhealthy", unhealthy)
                st.metric("Health Rate", f"{health_rate:.1f}%")
                st.progress(health_rate / 100)
                
                # Delete button
                if st.button(f"🗑️ Delete", key=f"del_{det_id}"):
                    db.delete_detection(det_id)
                    st.success("Deleted successfully!")
                    st.rerun()

def show_old_upload_page():
    # Header
    st.markdown('<h1 class="main-header">🌴 Palm Health Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">AI-Powered Aerial Palm Plantation Health Monitoring System</p>', unsafe_allow_html=True)
    
    # Sidebar for settings and info
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Settings")
        
        # Detection settings
        st.markdown("---")
        st.markdown("### 🎯 Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=0.95,
            value=0.05,
            step=0.01,
            help="Lower values detect more palms but may include false positives"
        )
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📊 Session Statistics")
        if 'total_images_processed' not in st.session_state:
            st.session_state.total_images_processed = 0
        if 'total_palms_detected' not in st.session_state:
            st.session_state.total_palms_detected = 0
        if 'total_healthy' not in st.session_state:
            st.session_state.total_healthy = 0
        if 'total_unhealthy' not in st.session_state:
            st.session_state.total_unhealthy = 0
        
        st.metric("Images Processed", st.session_state.total_images_processed)
        st.metric("Total Palms Detected", st.session_state.total_palms_detected)
        st.metric("Healthy Palms", st.session_state.total_healthy, delta_color="normal")
        st.metric("Unhealthy Palms", st.session_state.total_unhealthy, delta_color="inverse")
        
        # About
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This dashboard uses advanced AI to detect and classify palm health from aerial imagery.
        
        **Features:**
        - Real-time detection
        - Individual tree analysis
        - Health statistics
        - Export capabilities
        """)
    
    # Load model
    model, model_path = load_model()
    
    if model is None:
        st.error("❌ Failed to load the trained model. Please check if the model file exists.")
        st.stop()
    
    # Main dashboard tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Analytics Dashboard", "📖 User Guide"])
    
    with tab1:
        # Upload section
        st.markdown("### 📤 Upload Palm Images")
        uploaded_files = st.file_uploader(
            "Upload aerial palm images for health analysis",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Supported formats: JPG, JPEG, PNG",
            key="tab1_uploader"
        )
        
        if uploaded_files:
            st.markdown(f"### 🔍 Analysis Results ({len(uploaded_files)} image(s))")
            
            # Aggregate statistics for all images
            all_summaries = []
            
            # Process each uploaded image
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"🌴 {uploaded_file.name}", expanded=(idx == 0)):
                    # Load and display original image
                    img = Image.open(uploaded_file)
                    
                    # Detection controls
                    col_btn1, col_btn2 = st.columns([1, 4])
                    with col_btn1:
                        analyze_btn = st.button(f"🔍 Analyze", key=f"analyze_{idx}")
                    
                    if analyze_btn or idx == 0:  # Auto-analyze first image
                        # AI-powered palm health detection
                        with st.spinner("🤖 Analyzing palm health with AI..."):
                            results = auto_detect_palms(model, img, confidence_threshold=confidence_threshold)
                        
                        # Create summary
                        summary = create_detection_summary(results)
                        
                        if summary and summary['total_palms'] > 0:
                            # Update session statistics
                            st.session_state.total_images_processed += 1
                            st.session_state.total_palms_detected += summary['total_palms']
                            st.session_state.total_healthy += summary['healthy_palms']
                            st.session_state.total_unhealthy += summary['unhealthy_palms']
                            
                            all_summaries.append(summary)
                            
                            # Main metrics in colored cards
                            st.markdown("#### 📈 Detection Overview")
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            
                            with metric_col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h2 style="color: #2E8B57; margin: 0;">🌴 {summary['total_palms']}</h2>
                                    <p style="margin: 0; color: #666;">Total Palms</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h2 style="color: #4CAF50; margin: 0;">✅ {summary['healthy_palms']}</h2>
                                    <p style="margin: 0; color: #666;">Healthy</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h2 style="color: #F44336; margin: 0;">⚠️ {summary['unhealthy_palms']}</h2>
                                    <p style="margin: 0; color: #666;">Unhealthy</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_col4:
                                health_rate = (summary['healthy_palms'] / summary['total_palms'] * 100) if summary['total_palms'] > 0 else 0
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h2 style="color: #FF9800; margin: 0;">📊 {health_rate:.1f}%</h2>
                                    <p style="margin: 0; color: #666;">Health Rate</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Display results in columns
                            col_img, col_chart = st.columns([3, 2])
                            
                            with col_img:
                                st.markdown("#### 🖼️ Detection Visualization")
                                # Display detection results
                                if results and len(results) > 0 and hasattr(results[0], 'plot'):
                                    try:
                                        res_plotted = results[0].plot()
                                        st.image(
                                            res_plotted, 
                                            caption=f"Detection Results - {uploaded_file.name}",
                                            use_container_width=True
                                        )
                                    except:
                                        st.image(img, caption=f"Original Image - {uploaded_file.name}", use_container_width=True)
                                else:
                                    st.image(img, caption=f"No Palms Detected - {uploaded_file.name}", use_container_width=True)
                            
                            with col_chart:
                                st.markdown("#### 📊 Health Distribution")
                                # Create pie chart
                                health_chart = create_health_chart(summary)
                                if health_chart:
                                    st.plotly_chart(health_chart, use_container_width=True)
                                
                                # Health metrics with progress bar
                                st.markdown("#### 🎯 Health Metrics")
                                st.markdown(f"**Overall Health Rate**")
                                st.progress(health_rate / 100)
                                st.markdown(f"**Confidence Score**")
                                st.progress(summary['avg_confidence'])
                                st.caption(f"Average: {summary['avg_confidence']:.1%}")
                            
                            # Individual Palm Analysis
                            st.markdown("---")
                            st.markdown("#### 🔬 Individual Palm Tree Analysis")
                            
                            # Crop individual palms
                            cropped_palms = crop_individual_palms(img, results)
                            
                            if cropped_palms:
                                # Filter options
                                filter_col1, filter_col2 = st.columns(2)
                                with filter_col1:
                                    show_filter = st.selectbox(
                                        "Filter by status:",
                                        ["All Palms", "Healthy Only", "Unhealthy Only"],
                                        key=f"filter_{idx}"
                                    )
                                
                                # Apply filter
                                filtered_palms = cropped_palms
                                if show_filter == "Healthy Only":
                                    filtered_palms = [p for p in cropped_palms if p['status'] == 'Healthy']
                                elif show_filter == "Unhealthy Only":
                                    filtered_palms = [p for p in cropped_palms if p['status'] == 'Unhealthy']
                                
                                st.caption(f"Showing {len(filtered_palms)} of {len(cropped_palms)} palms")
                                
                                # Display in grid
                                cols_per_row = 4
                                rows = (len(filtered_palms) + cols_per_row - 1) // cols_per_row
                                
                                for row in range(rows):
                                    cols = st.columns(cols_per_row)
                                    for col_idx in range(cols_per_row):
                                        palm_idx = row * cols_per_row + col_idx
                                        if palm_idx < len(filtered_palms):
                                            palm = filtered_palms[palm_idx]
                                            
                                            with cols[col_idx]:
                                                st.image(
                                                    palm['image'],
                                                    use_container_width=True
                                                )
                                                
                                                status_emoji = "✅" if palm['status'] == "Healthy" else "⚠️"
                                                confidence_color = "🟢" if palm['confidence'] > 0.7 else "🟡" if palm['confidence'] > 0.4 else "🔴"
                                                
                                                st.markdown(f"""
                                                <div class="palm-card">
                                                    <div style="text-align: center;">
                                                        <strong style="color: {palm['status_color']};">
                                                            {status_emoji} Palm {palm['id']}
                                                        </strong><br>
                                                        <span style="font-size: 0.9rem;">
                                                            {palm['status']}<br>
                                                            {confidence_color} {palm['confidence']:.1%}
                                                        </span>
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                            
                            # Detailed table
                            if summary['detections']:
                                st.markdown("---")
                                st.markdown("#### 📋 Detailed Detection Table")
                                
                                detection_data = []
                                for det in summary['detections']:
                                    detection_data.append({
                                        'ID': det['id'],
                                        'Status': det['status'],
                                        'Confidence': f"{det['confidence']:.1%}",
                                        'Class': det['class_name']
                                    })
                                
                                df = pd.DataFrame(detection_data)
                                
                                def style_status(val):
                                    if val == 'Healthy':
                                        return 'background-color: #d4edda; color: #155724'
                                    else:
                                        return 'background-color: #f8d7da; color: #721c24'
                                
                                styled_df = df.style.applymap(style_status, subset=['Status'])
                                st.dataframe(styled_df, use_container_width=True)
                            
                            # Individual Palm Crops Section
                            st.markdown("---")
                            st.markdown("#### 🌴 Individual Palm Tree Analysis")
                            
                            # Crop individual palms and classify with Faster R-CNN
                            cropped_palms = crop_individual_palms(img, results, faster_rcnn_model)
                            
                            if cropped_palms:
                                # Display cropped palms in a grid
                                cols_per_row = 3
                                rows = (len(cropped_palms) + cols_per_row - 1) // cols_per_row
                                
                                for row in range(rows):
                                    cols = st.columns(cols_per_row)
                                    for col_idx in range(cols_per_row):
                                        palm_idx = row * cols_per_row + col_idx
                                        if palm_idx < len(cropped_palms):
                                            palm = cropped_palms[palm_idx]
                                            
                                            with cols[col_idx]:
                                                # Display cropped palm image
                                                st.image(
                                                    palm['image'],
                                                    caption=f"Palm {palm['id']}: {palm['status']}",
                                                    use_container_width=True
                                                )
                                                
                                                # Status badge with color
                                                status_emoji = "✅" if palm['status'] == "Healthy" else "⚠️"
                                                confidence_color = "🟢" if palm['confidence'] > 0.7 else "🟡" if palm['confidence'] > 0.4 else "🔴"
                                                
                                                st.markdown(f"""
                                                <div style="text-align: center; padding: 5px; background-color: {palm['status_color']}20; border-radius: 5px; margin: 5px 0;">
                                                    <strong>{status_emoji} {palm['status']}</strong><br>
                                                    {confidence_color} {palm['confidence']:.1%} confidence
                                                </div>
                                                """, unsafe_allow_html=True)
                        
                        else:
                            st.warning("🔍 No palms detected in this image. Try adjusting the confidence threshold in the sidebar.")
                            st.image(img, caption=f"Original Image - {uploaded_file.name}", use_container_width=True)
        
        else:
            # Welcome message with nice design
            st.markdown("""
            <div class="info-box">
                <h3 style="margin-top: 0;">🎯 Welcome to Palm Health Detection Dashboard!</h3>
                <p><strong>How to use:</strong></p>
                <ol>
                    <li>📤 Upload aerial palm images using the file uploader above</li>
                    <li>⚙️ Adjust detection settings in the sidebar if needed</li>
                    <li>🤖 The system will automatically analyze your images</li>
                    <li>📊 View detailed health statistics and individual tree analysis</li>
                </ol>
                
                <p><strong>🤖 AI Capabilities:</strong></p>
                <ul>
                    <li>✅ High accuracy rate with advanced deep learning</li>
                    <li>🎯 Automatic detection optimization for different image types</li>
                    <li>🌴 Individual tree detection in dense plantations</li>
                    <li>📈 Real-time health classification and comprehensive analytics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample images section
            st.markdown("### 📸 Sample Analysis")
            col_ex1, col_ex2, col_ex3 = st.columns(3)
            
            with col_ex1:
                st.markdown("""
                <div class="palm-card">
                    <h4 style="color: #2E8B57;">🌴 Dense Plantation</h4>
                    <p style="font-size: 0.9rem;">Analyzes individual trees in crowded plantations</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_ex2:
                st.markdown("""
                <div class="palm-card">
                    <h4 style="color: #2E8B57;">🔍 Health Detection</h4>
                    <p style="font-size: 0.9rem;">Identifies healthy vs unhealthy palms automatically</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_ex3:
                st.markdown("""
                <div class="palm-card">
                    <h4 style="color: #2E8B57;">📊 Analytics</h4>
                    <p style="font-size: 0.9rem;">Provides comprehensive health statistics</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Analytics Dashboard")
        
        if st.session_state.total_images_processed == 0:
            st.info("📈 Upload and analyze images to see analytics dashboard")
        else:
            # Overall statistics
            st.markdown("#### 🌍 Overall Plantation Health")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Images", st.session_state.total_images_processed)
            with metric_col2:
                st.metric("Total Palms", st.session_state.total_palms_detected)
            with metric_col3:
                overall_health_rate = (st.session_state.total_healthy / st.session_state.total_palms_detected * 100) if st.session_state.total_palms_detected > 0 else 0
                st.metric("Health Rate", f"{overall_health_rate:.1f}%")
            with metric_col4:
                avg_palms_per_image = st.session_state.total_palms_detected / st.session_state.total_images_processed if st.session_state.total_images_processed > 0 else 0
                st.metric("Avg Palms/Image", f"{avg_palms_per_image:.0f}")
            
            # Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Overall health distribution
                fig_overall = go.Figure(data=[go.Pie(
                    labels=['Healthy', 'Unhealthy'],
                    values=[st.session_state.total_healthy, st.session_state.total_unhealthy],
                    hole=0.4,
                    marker_colors=['#4CAF50', '#F44336']
                )])
                fig_overall.update_layout(title="Overall Health Distribution", height=400)
                st.plotly_chart(fig_overall, use_container_width=True)
            
            with col_chart2:
                # Bar chart
                fig_bar = go.Figure(data=[
                    go.Bar(name='Healthy', x=['Palms'], y=[st.session_state.total_healthy], marker_color='#4CAF50'),
                    go.Bar(name='Unhealthy', x=['Palms'], y=[st.session_state.total_unhealthy], marker_color='#F44336')
                ])
                fig_bar.update_layout(title="Health Comparison", height=400, barmode='group')
                st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.markdown("### 📖 User Guide")
        
        st.markdown("""
        <div class="info-box">
            <h4>🚀 Getting Started</h4>
            <p>This dashboard provides AI-powered palm health detection from aerial imagery.</p>
        </div>
        
        <div class="info-box">
            <h4>⚙️ Settings Guide</h4>
            <ul>
                <li><strong>Confidence Threshold:</strong> Lower values detect more palms but may include false positives. Recommended: 0.05-0.15</li>
                <li><strong>Tile-Based Detection:</strong> Enable for very dense plantations or when standard detection misses trees</li>
            </ul>
        </div>
        
        <div class="info-box">
            <h4>📊 Understanding Results</h4>
            <ul>
                <li><strong>Green boxes:</strong> Healthy palms</li>
                <li><strong>Red boxes:</strong> Unhealthy palms requiring attention</li>
                <li><strong>Confidence score:</strong> AI's certainty about the classification (higher is better)</li>
                <li><strong>Health Rate:</strong> Percentage of healthy palms in the plantation</li>
            </ul>
        </div>
        
        <div class="warning-box">
            <h4>⚠️ Best Practices</h4>
            <ul>
                <li>Use high-resolution aerial images for best results</li>
                <li>Ensure good lighting conditions in images</li>
                <li>Avoid heavily shadowed or cloudy images</li>
                <li>For dense plantations, enable tile-based detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**🌴 Palm Health Detection System**")
    with footer_col2:
        st.markdown("**📍 Aerial Image Analysis**")
    with footer_col3:
        st.markdown(f"**📅 Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
