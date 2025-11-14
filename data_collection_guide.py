# Palm Health Detection - Data Collection Guide
# =====================================================

"""
SOLUTION: Expand training data with aerial/plantation images

Current Problem:
- Model trained on individual palm images (close-up views)
- Testing on aerial plantation images (many small palms)
- Domain mismatch causing poor detection

Data Collection Strategy:
"""

import os
import requests
from pathlib import Path

def download_aerial_palm_datasets():
    """
    Guide for collecting aerial palm plantation datasets
    """
    
    print("🌴 AERIAL PALM DATA COLLECTION GUIDE")
    print("=" * 50)
    
    print("\n1. ROBOFLOW DATASETS:")
    print("   - Search: 'aerial palm plantation'")
    print("   - Search: 'oil palm plantation aerial'") 
    print("   - Search: 'date palm aerial view'")
    print("   - URL: https://roboflow.com/")
    
    print("\n2. GOOGLE EARTH ENGINE:")
    print("   - Satellite imagery of palm plantations")
    print("   - URL: https://earthengine.google.com/")
    
    print("\n3. KAGGLE DATASETS:")
    print("   - Palm Oil Plantation Dataset")
    print("   - Aerial Agriculture Datasets")
    print("   - URL: https://www.kaggle.com/")
    
    print("\n4. MANUAL COLLECTION:")
    print("   - Google Maps satellite view")
    print("   - Drone footage from YouTube")
    print("   - Agricultural research papers")
    
    print("\n5. DATA AUGMENTATION FOR AERIAL VIEWS:")
    print("   - Crop sections from large plantation images")
    print("   - Simulate different altitudes")
    print("   - Various lighting conditions")

def create_data_collection_plan():
    """
    Create a systematic plan for collecting aerial palm data
    """
    
    plan = {
        "target_images": 300,  # Add 300 aerial images
        "image_types": {
            "high_altitude": 100,   # Very high aerial views
            "medium_altitude": 100, # Medium aerial views  
            "low_altitude": 100,    # Lower aerial views
        },
        "palm_densities": {
            "sparse": 50,     # Few palms per image
            "medium": 150,    # Moderate density
            "dense": 100,     # High density (plantation style)
        },
        "lighting_conditions": {
            "bright_sunny": 100,
            "overcast": 100, 
            "golden_hour": 100,
        }
    }
    
    print("\n📋 DATA COLLECTION PLAN:")
    print("=" * 30)
    for category, details in plan.items():
        if isinstance(details, dict):
            print(f"\n{category.upper()}:")
            for subcategory, count in details.items():
                print(f"  - {subcategory}: {count} images")
        else:
            print(f"{category}: {details}")
    
    return plan

def prepare_retraining_structure():
    """
    Set up folder structure for new training data
    """
    
    base_dir = Path("datasets/palms_yolo_v2")
    
    folders = [
        "train/images",
        "train/labels", 
        "valid/images",
        "valid/labels",
        "test/images", 
        "test/labels"
    ]
    
    print("\n📁 CREATING NEW DATASET STRUCTURE:")
    print("=" * 40)
    
    for folder in folders:
        folder_path = base_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {folder_path}")
    
    # Create updated dataset.yaml
    yaml_content = f"""
# Updated Palm Health Detection Dataset v2
# Includes aerial plantation images

names:
  0: PalmAnom  # Anomalous/Unhealthy Palm
  1: PalmSan   # Healthy Palm

nc: 2
path: {base_dir.absolute()}

train: train/images
val: valid/images  
test: test/images

# Training settings for aerial detection
train_settings:
  imgsz: 1024        # Larger image size for aerial detection
  batch: 16          # Adjust based on GPU memory
  epochs: 100        # More epochs for complex aerial patterns
  patience: 20       # Early stopping patience
  
# Augmentation for aerial views
augmentation:
  mosaic: 0.5        # Mosaic augmentation
  copy_paste: 0.1    # Copy-paste augmentation  
  mixup: 0.1         # Mixup augmentation
"""
    
    yaml_path = base_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created: {yaml_path}")
    
    return base_dir

if __name__ == "__main__":
    download_aerial_palm_datasets()
    create_data_collection_plan() 
    prepare_retraining_structure()
    
    print("\n🎯 NEXT STEPS:")
    print("=" * 15)
    print("1. Collect aerial palm images using the guide above")
    print("2. Label the images using tools like LabelImg or Roboflow")
    print("3. Add images to datasets/palms_yolo_v2/ structure")
    print("4. Run train.py with the new dataset")
    print("5. Test on aerial images")