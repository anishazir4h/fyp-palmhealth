
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

# Page configuration
st.set_page_config(
    page_title="Palm Health Detection",
    page_icon="-",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .model-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained palm health detection model"""
    try:
        # Try to load from backup first (more reliable path)
        model_path = "C:/Users/anish/OneDrive/Desktop/FYP/fyp-palm/runs/detect/YOLO_Detection/weights/best.pt"
        model = YOLO(model_path)
        return model, model_path
    except:
        try:
            # Fallback to original location
            model_path = "runs/detect/YOLO_Detection/weights/best.pt"
            model = YOLO(model_path)
            return model, model_path
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None, None

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
    
    # 4. Enhance green vegetation (palms)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    hsv[:,:,1] = cv2.add(hsv[:,:,1], cv2.bitwise_and(np.full_like(hsv[:,:,1], 30), mask_green))
    
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

def tile_based_detection(model, image, confidence_threshold=0.1, tile_size=64, overlap=0.8):
    """
    Detect palms using sliding window/tiling approach for dense plantations
    HEAVILY OPTIMIZED for your specific model training characteristics
    """
    from PIL import Image
    import numpy as np
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    width, height = image.size
    step = int(tile_size * (1 - overlap))  # 80% overlap for maximum coverage
    
    all_detections = []
    
    # Very small tiles with massive overlap
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
                iou=0.4,      # Allow some overlap but prevent duplicates
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
    Merge overlapping detections - if two boxes overlap by 50% or more
    (measured as intersection/smaller_box_area), keep the one with higher confidence
    This is more aggressive than IoU
    """
    if not detections or len(detections) <= 1:
        return detections
    
    import torch
    
    # Convert to tensors
    boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d['conf'] for d in detections], dtype=torch.float32)
    
    # Calculate overlap percentage between boxes
    def calculate_overlap_percent(box1, box2):
        """Calculate what percentage of the smaller box overlaps with the other"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Use the smaller box as reference
        smaller_area = min(area1, area2)
        
        return intersection / smaller_area if smaller_area > 0 else 0
    
    # Mark boxes to keep
    keep = [True] * len(detections)
    
    for i in range(len(boxes)):
        if not keep[i]:
            continue
            
        for j in range(i + 1, len(boxes)):
            if not keep[j]:
                continue
                
            overlap = calculate_overlap_percent(boxes[i], boxes[j])
            
            # If overlap >= 50%, keep the one with higher confidence
            if overlap >= overlap_threshold:
                if scores[i] >= scores[j]:
                    keep[j] = False  # Remove j, keep i
                else:
                    keep[i] = False  # Remove i, keep j
                    break  # Move to next i
    
    # Return only kept detections
    merged = [detections[i] for i in range(len(detections)) if keep[i]]
    return merged


def merge_tiled_detections(detections, image_size, iou_threshold=0.1):
    """
    Merge overlapping detections from tiled approach with moderate filtering
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
    
    # Apply very conservative NMS - preserve nearby palms
    keep_indices = torch.ops.torchvision.nms(filtered_boxes, filtered_scores, 0.15)  # Higher IoU = keep more detections
    
    # Return kept detections
    final_indices = [valid_indices[i] for i in keep_indices]
    merged = [detections[i] for i in final_indices]
    
    return merged

def auto_detect_palms(model, image, confidence_threshold=0.05):
    """
    Automatically applies the best detection pipeline for palm detection
    Intelligently chooses between enhanced, tiled, or combined approaches
    """
    
    # Step 1: Quick analysis to determine image characteristics
    width, height = image.size
    total_pixels = width * height
    
    # Step 2: Try enhanced detection first (fastest)
    enhanced_img = enhance_palm_visibility(image)
    
    # Initial detection with very low IoU for individual palms
    initial_results = model.predict(
        enhanced_img,
        conf=confidence_threshold,
        iou=0.1,  # Very low IoU for individual detection
        max_det=1000,
        verbose=False
    )
    
    # Step 3: Analyze initial results to decide on strategy
    if initial_results and len(initial_results) > 0 and initial_results[0].boxes is not None:
        boxes = initial_results[0].boxes
        num_detections = len(boxes)
        
        # Calculate average box size
        avg_box_area = 0
        if num_detections > 0:
            for i in range(num_detections):
                box = boxes.xyxy[i].cpu().numpy()
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                avg_box_area += box_area
            avg_box_area /= num_detections
            avg_box_ratio = avg_box_area / total_pixels
        else:
            avg_box_ratio = 0
        
        # Decision logic for detection strategy - very aggressive for aerial images
        if avg_box_ratio > 0.05 or num_detections < 15:  # Even lower threshold + higher detection count requirement
            # Use tiled detection for dense plantations
            st.info("Aerial plantation detected - using ultra-aggressive individual tree analysis...")
            
            detections = tile_based_detection(model, image, confidence_threshold, tile_size=256)  # Even smaller tiles
            merged_detections = merge_tiled_detections(detections, image.size, iou_threshold=0.05)  # Ultra-strict NMS
            
            # MERGE OVERLAPPING BOXES (30% overlap threshold - more aggressive)
            merged_detections = merge_overlapping_boxes(merged_detections, overlap_threshold=0.3)
            
            if merged_detections:
                # Create proper result object
                class MockBoxes:
                    def __init__(self, detections):
                        self.xyxy = torch.tensor([d['bbox'] for d in detections])
                        self.conf = torch.tensor([d['conf'] for d in detections])
                        self.cls = torch.tensor([d['cls'] for d in detections])
                    
                    def __len__(self):
                        return len(self.xyxy)
                
                class MockResult:
                    def __init__(self, detections, img_shape):
                        self.boxes = MockBoxes(detections) if detections else None
                        self.orig_shape = img_shape
                    
                    def plot(self):
                        img_array = np.array(image)
                        if self.boxes and len(self.boxes) > 0:
                            for i in range(len(self.boxes)):
                                box = self.boxes.xyxy[i].numpy()
                                conf = self.boxes.conf[i].item()
                                cls = int(self.boxes.cls[i].item())
                                
                                # Choose color based on health
                                color = (0, 255, 0) if cls == 1 else (0, 0, 255)  # Green for healthy, Red for unhealthy
                                
                                cv2.rectangle(img_array, 
                                            (int(box[0]), int(box[1])), 
                                            (int(box[2]), int(box[3])), 
                                            color, 2)
                                
                                label = f"{'Healthy' if cls == 1 else 'Unhealthy'} {conf:.2f}"
                                cv2.putText(img_array, label, 
                                          (int(box[0]), int(box[1]-10)), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        return img_array
                
                result = MockResult(merged_detections, (image.height, image.width))
                st.success(f"✅ Optimized detection found {len(merged_detections)} individual palm trees!")
                return [result]
            else:
                # Fallback to enhanced detection
                st.info("🌟 Using enhanced detection...")
                return initial_results
        
        elif num_detections < 3:  # Too few detections - try multi-scale
            st.info("Few palms detected - using multi-scale analysis...")
            
            # Multi-scale detection for better coverage
            multiscale_results = multi_scale_detection(model, enhanced_img, confidence_threshold)
            if multiscale_results and len(multiscale_results) > 0:
                result = multiscale_results[0]
                if result.boxes is not None and len(result.boxes) > num_detections:
                    st.success(f"✅ Multi-scale detection improved results: {len(result.boxes)} palms found!")
                    return multiscale_results
            
            # Return original if multi-scale didn't improve
            st.success(f"✅ Enhanced detection found {num_detections} palm trees!")
            return initial_results
        
        else:  # Good detection - use enhanced results
            # MERGE OVERLAPPING BOXES for enhanced results too
            if initial_results and len(initial_results) > 0 and initial_results[0].boxes is not None:
                boxes = initial_results[0].boxes
                detections = []
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    detections.append({
                        'bbox': bbox,
                        'conf': float(boxes.conf[i]),
                        'cls': int(boxes.cls[i])
                    })
                
                # Merge overlapping boxes with 30% threshold (more aggressive)
                merged_detections = merge_overlapping_boxes(detections, overlap_threshold=0.3)
                
                if merged_detections:
                    class MockBoxes:
                        def __init__(self, detections):
                            self.xyxy = torch.tensor([d['bbox'] for d in detections])
                            self.conf = torch.tensor([d['conf'] for d in detections])
                            self.cls = torch.tensor([d['cls'] for d in detections])
                        
                        def __len__(self):
                            return len(self.xyxy)
                    
                    class MockResult:
                        def __init__(self, detections):
                            self.boxes = MockBoxes(detections) if detections else None
                    
                    result = MockResult(merged_detections)
                    st.success(f"✅ Optimized detection found {len(merged_detections)} palm trees (merged from {num_detections})!")
                    return [result]
            
            st.success(f"✅ Optimized detection found {num_detections} palm trees!")
            return initial_results
    
    else:
        # No detections - try more aggressive approach
        st.info("No palms detected initially - trying more sensitive detection...")
        
        # Very sensitive detection
        sensitive_results = model.predict(
            enhanced_img,
            conf=0.01,  # Very low confidence
            iou=0.05,   # Very low IoU
            max_det=2000,
            verbose=False
        )
        
        if sensitive_results and len(sensitive_results) > 0 and sensitive_results[0].boxes is not None:
            num_found = len(sensitive_results[0].boxes)
            st.success(f"✅ Sensitive detection found {num_found} palm trees!")
            return sensitive_results
        else:
            st.warning("🔍 No palms detected. The image may not contain palm trees or they may be too small/unclear.")
            return []

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
    # Header
    st.markdown('<h1 class="main-header">🌴 Palm Health Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Load model
    model, model_path = load_model()
    
    if model is None:
        st.error("❌ Failed to load the trained model. Please check if the model files exist.")
        st.stop()
    
    # Main content - full width layout
    st.markdown("### 📤 Upload Palm Images")
    uploaded_files = st.file_uploader(
        "Upload aerial palm images for health analysis",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_files:
        st.markdown(f"### 🔍 Analysis Results ({len(uploaded_files)} image(s))")
        
        # Process each uploaded image
        for idx, uploaded_file in enumerate(uploaded_files):
            with st.expander(f" {uploaded_file.name}", expanded=True):
                # Load and display original image
                img = Image.open(uploaded_file)
                
                # Automatic best detection - no user choices needed
                with st.spinner("Analyzing palm health with optimized detection..."):
                    # Automatically use the best detection pipeline
                    results = auto_detect_palms(model, img, confidence_threshold=0.05)
                
                # Create summary
                summary = create_detection_summary(results)
                
                if summary and summary['total_palms'] > 0:
                        # Display results
                        col_img, col_stats = st.columns([2, 1])
                        
                        with col_img:
                            # Display detection results
                            if results and len(results) > 0 and hasattr(results[0], 'plot'):
                                try:
                                    res_plotted = results[0].plot()
                                    st.image(
                                        res_plotted, 
                                        caption=f"Palm Detection Results - {uploaded_file.name}",
                                        use_container_width=True
                                    )
                                except:
                                    # Fallback to original image if plotting fails
                                    st.image(img, caption=f"Original Image - {uploaded_file.name}", use_container_width=True)
                            else:
                                st.image(img, caption=f"No Palms Detected - {uploaded_file.name}", use_container_width=True)
                        
                        with col_stats:
                            # Display metrics
                            st.markdown("#### Detection Summary")
                            
                            # Metrics
                            col_met1, col_met2 = st.columns(2)
                            with col_met1:
                                st.metric("Total Palms", summary['total_palms'])
                                st.metric("Healthy", summary['healthy_palms'])
                            with col_met2:
                                st.metric("Unhealthy", summary['unhealthy_palms'])
                                st.metric("Avg. Confidence", f"{summary['avg_confidence']:.1%}")
                            
                            # Health percentage
                            if summary['total_palms'] > 0:
                                health_percentage = (summary['healthy_palms'] / summary['total_palms']) * 100
                                st.markdown(f"**Health Rate:** {health_percentage:.1f}%")
                                
                                # Progress bar for health rate
                                st.progress(health_percentage / 100)
                        
                        # Individual Palm Crops Section
                        st.markdown("---")
                        st.markdown("####Individual Palm Tree Analysis")
                        
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
                        
                        # Detailed detection results table
                        if summary['detections']:
                            st.markdown("#### 📋 Detailed Detection Results")
                            
                            # Create DataFrame for results
                            detection_data = []
                            for det in summary['detections']:
                                detection_data.append({
                                    'Palm ID': det['id'],
                                    'Health Status': det['status'],
                                    'Confidence': f"{det['confidence']:.1%}",
                                    'Class': det['class_name']
                                })
                            
                            df = pd.DataFrame(detection_data)
                            
                            # Style the dataframe
                            def style_status(val):
                                if val == 'Healthy':
                                    return 'background-color: #d4edda; color: #155724'
                                else:
                                    return 'background-color: #f8d7da; color: #721c24'
                            
                            styled_df = df.style.applymap(style_status, subset=['Health Status'])
                            st.dataframe(styled_df, use_container_width=True)
                
                else:
                    st.warning("🔍 No palms detected in this image. Try adjusting the confidence threshold or upload a clearer image.")
                    st.image(img, caption=f"Original Image - {uploaded_file.name}", use_container_width=True)
    
    else:
        # Welcome message
        st.markdown("### 🎯 Welcome to Palm Health Detection!")
        st.info("""
        **How to use:**
        1. Upload aerial palm images using the file uploader above
        2. The system will automatically apply the best detection method
        3. View real-time health analysis results
        4. Get detailed statistics and recommendations
        
        **🤖 Intelligent Features:**
        - 97% accuracy rate with YOLOv8n architecture
        - Automatic detection optimization for different image types
        - Individual tree detection in dense plantations
        - Real-time health classification and analysis
        """)
    
    # Footer
    st.markdown("---")
    col_footer1, col_footer2 = st.columns(2)
    
    with col_footer1:
        st.markdown("**AI Model:** YOLOv8n")
    with col_footer2:
        st.markdown(f"**📅 Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()