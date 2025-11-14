"""
Visual comparison of NMS settings to demonstrate duplicate reduction
Run this to see before/after comparison
"""

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

def draw_boxes_on_image(img, results, title=""):
    """Draw bounding boxes on image with count"""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        count = len(boxes)
        
        # Draw boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            # Green box
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        
        # Add title with count
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        text = f"{title}: {count} palms"
        draw.text((10, 10), text, fill="yellow", font=font, stroke_width=2, stroke_fill="black")
    
    return img_copy

def compare_nms_settings(image_path):
    """Compare different NMS IoU thresholds"""
    
    print("🌴 Palm Detection - NMS Comparison Tool")
    print("="*60)
    
    # Load model
    try:
        model_path = "C:/Users/anish/OneDrive/Desktop/FYP/fyp-palm/runs/detect/YOLO_Detection/weights/best.pt"
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load image
    try:
        img = Image.open(image_path)
        print(f"✅ Image loaded: {img.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    print("\n" + "="*60)
    print("Running detections with different NMS settings...")
    print("="*60)
    
    # Test different IoU values
    test_cases = [
        ("OLD (IoU=0.45)", 0.45),
        ("NEW (IoU=0.30)", 0.30),
        ("AGGRESSIVE (IoU=0.20)", 0.20),
    ]
    
    results_images = []
    
    for title, iou_val in test_cases:
        print(f"\nTesting {title}...")
        
        # Run detection
        results = model.predict(
            img,
            conf=0.05,
            iou=iou_val,
            max_det=300,
            verbose=False
        )
        
        # Count detections
        if results and len(results) > 0 and results[0].boxes is not None:
            count = len(results[0].boxes)
            print(f"  → Detected: {count} palms")
        else:
            count = 0
            print(f"  → Detected: 0 palms")
        
        # Draw boxes on image
        result_img = draw_boxes_on_image(img, results, title)
        results_images.append((title, result_img, count))
    
    # Save comparison images
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Saving comparison images...")
    print("="*60)
    
    for title, result_img, count in results_images:
        filename = f"{title.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.jpg"
        filepath = os.path.join(output_dir, filename)
        result_img.save(filepath)
        print(f"✅ Saved: {filepath}")
    
    # Create side-by-side comparison
    if len(results_images) >= 2:
        print("\nCreating side-by-side comparison...")
        
        # Resize images for side-by-side
        width, height = results_images[0][1].size
        new_width = width // 2
        new_height = height // 2
        
        # Create canvas for 3 images
        comparison = Image.new('RGB', (new_width * 3, new_height))
        
        for i, (title, result_img, count) in enumerate(results_images):
            resized = result_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            comparison.paste(resized, (i * new_width, 0))
        
        comparison_path = os.path.join(output_dir, "COMPARISON_SIDE_BY_SIDE.jpg")
        comparison.save(comparison_path)
        print(f"✅ Saved comparison: {comparison_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for title, _, count in results_images:
        print(f"{title:30} → {count:3} palms")
    
    old_count = results_images[0][2]
    new_count = results_images[1][2]
    reduction = ((old_count - new_count) / old_count * 100) if old_count > 0 else 0
    
    print("\n" + "="*60)
    print(f"IMPROVEMENT: {reduction:.1f}% reduction in detections")
    print("="*60)
    
    if reduction > 0:
        print("✅ The NEW setting removes duplicates more effectively!")
    elif reduction < 0:
        print("⚠️  The NEW setting may be too aggressive (missing some palms)")
    else:
        print("ℹ️  No change - image may have few duplicates")
    
    print("\n📁 All results saved to:", os.path.abspath(output_dir))
    print("\n✨ Open the images to see the visual difference!")

if __name__ == "__main__":
    print("\n🌴 NMS Comparison Tool - Visual Before/After")
    print("="*60)
    
    # Get image path from user
    image_path = input("\nEnter path to your test image: ").strip()
    
    # Remove quotes if user copy-pasted
    image_path = image_path.strip('"').strip("'")
    
    if not os.path.exists(image_path):
        print(f"\n❌ File not found: {image_path}")
        print("\nExample paths:")
        print('  "C:/Users/anish/OneDrive/Desktop/FYP/fyp-palm/uploaded_images/your_image.jpg"')
        print('  "datasets/palms_yolo/test/images/some_image.jpg"')
    else:
        compare_nms_settings(image_path)
