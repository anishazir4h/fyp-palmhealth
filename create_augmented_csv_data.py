"""
Create augmented CSV data matching YOLO structure
- train_aug: Original + augmented (like YOLO train_aug with 858 images)
- valid_aug: Copy of valid (22 images, no augmentation)
- test_aug: Copy of test (64 images, no augmentation)
"""

import os
import cv2
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm


def process_split(split_name, should_augment=True):
    """Process a dataset split (train, valid, or test)"""
    
    print(f"\n{'='*80}")
    print(f"📊 Processing {split_name.upper()} split")
    print(f"{'='*80}")
    
    # Paths
    original_img_dir = f"datasets/{split_name}"
    original_csv = f"datasets/{split_name}/_annotations.csv"
    aug_img_dir = f"datasets/{split_name}_aug"
    aug_csv = f"datasets/{split_name}_aug/_annotations.csv"
    
    # Create augmented directory
    os.makedirs(aug_img_dir, exist_ok=True)
    
    print(f"📂 Setup:")
    print(f"   Source: {original_img_dir}")
    print(f"   Target: {aug_img_dir}")
    print(f"   Augmentation: {'YES' if should_augment else 'NO (copy only)'}")
    
    # Read original CSV
    print(f"📊 Reading CSV...")
    df = pd.read_csv(original_csv)
    original_images = df['filename'].unique()
    print(f"   Images: {len(original_images)}")
    print(f"   Annotations: {len(df)}")
    
    all_annotations = []
    
    # Copy original images
    print(f"📋 Copying original images...")
    for img_file in tqdm(original_images, desc="Copying"):
        src_path = os.path.join(original_img_dir, img_file)
        dst_path = os.path.join(aug_img_dir, img_file)
        
        try:
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)
                img_annotations = df[df['filename'] == img_file]
                all_annotations.append(img_annotations)
        except Exception as e:
            print(f"   ⚠️ Error copying {img_file}: {e}")
    
    print(f"   ✅ Copied {len(original_images)} images")
    
    # Create augmented versions (only for train)
    augmented_count = 0
    
    if should_augment:
        print(f"🔄 Creating augmented versions...")
        
        # Create augmented versions to match YOLO: ~858 images total
        # 338 original + need ~520 augmented = ~858 total
        # Strategy: Create 1-2 augmented versions randomly (avg 1.5x = 507 aug = 845 total, close to 858)
        augmentations_per_image = 1  # Create only 1 augmented version per image to better match YOLO
        
        # Create augmented versions with randomness to match YOLO's ~858 total
        # For ~60% of images: create 2 augs, for rest: create 1 aug
        # This gives: 338 original + (~200*2 + ~138*1) = 338 + 538 = ~876 total (close to 858)
        for img_file in tqdm(original_images, desc="Creating augmentations"):
            # Decide how many augmentations for this image (1 or 2)
            num_augs = 2 if np.random.random() < 0.6 else 1
            
            for aug_idx in range(num_augs):
                try:
                    # Read image
                    img_path = os.path.join(original_img_dir, img_file)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    # Get annotations
                    img_annotations = df[df['filename'] == img_file].copy()
                    
                    # Apply augmentation
                    aug_image = image.copy()
                    aug_annotations = img_annotations.copy()
                    
                    # Random horizontal flip (50%)
                    if np.random.random() > 0.5:
                        aug_image = cv2.flip(aug_image, 1)
                        img_width = image.shape[1]
                        aug_annotations['xmin_new'] = img_width - img_annotations['xmax']
                        aug_annotations['xmax_new'] = img_width - img_annotations['xmin']
                        aug_annotations['xmin'] = aug_annotations['xmin_new']
                        aug_annotations['xmax'] = aug_annotations['xmax_new']
                        aug_annotations.drop(['xmin_new', 'xmax_new'], axis=1, inplace=True)
                    
                    # Random brightness (60%)
                    if np.random.random() > 0.4:
                        brightness = np.random.uniform(0.7, 1.3)
                        aug_image = cv2.convertScaleAbs(aug_image, alpha=brightness, beta=0)
                    
                    # Random contrast (60%)
                    if np.random.random() > 0.4:
                        contrast = np.random.uniform(0.8, 1.2)
                        aug_image = cv2.convertScaleAbs(aug_image, alpha=contrast, beta=0)
                    
                    # Random noise (30%)
                    if np.random.random() > 0.7:
                        noise = np.random.normal(0, 8, aug_image.shape).astype(np.int16)
                        aug_image = np.clip(aug_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    
                    # Random rotation (40%)
                    if np.random.random() > 0.6:
                        angle = np.random.uniform(-10, 10)
                        h, w = aug_image.shape[:2]
                        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                        aug_image = cv2.warpAffine(aug_image, M, (w, h))
                    
                    # Save augmented image
                    name_base = os.path.splitext(img_file)[0]
                    ext = os.path.splitext(img_file)[1]
                    aug_img_name = f"{name_base}_aug{aug_idx}{ext}"
                    aug_img_path = os.path.join(aug_img_dir, aug_img_name)
                    cv2.imwrite(aug_img_path, aug_image)
                    
                    # Update filename in annotations
                    aug_annotations['filename'] = aug_img_name
                    all_annotations.append(aug_annotations)
                    
                    augmented_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Error augmenting {img_file}: {e}")
                    continue
        
        print(f"   ✅ Created {augmented_count} augmented images")
    
    # Save combined CSV
    print(f"💾 Saving CSV...")
    combined_df = pd.concat(all_annotations, ignore_index=True)
    combined_df.to_csv(aug_csv, index=False)
    
    total_images = len(original_images) + augmented_count
    total_annotations = len(combined_df)
    
    print(f"\n✅ {split_name.upper()} COMPLETE:")
    print(f"   Original: {len(original_images)} images")
    print(f"   Augmented: {augmented_count} images")
    print(f"   Total: {total_images} images")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Output: {aug_img_dir}")
    
    return total_images, total_annotations


def main():
    """Create augmented CSV data for all splits"""
    
    print("=" * 80)
    print("🎯 CREATING AUGMENTED CSV DATA - MATCHING YOLO STRUCTURE")
    print("=" * 80)
    print("\nTarget structure (matching palms_yolo):")
    print("  - YOLO train_aug: 858 images (338 original + 520 augmented)")
    print("  - YOLO valid: 22 images (no augmentation)")
    print("  - YOLO test: 64 images (no augmentation)")
    print("\nWe will create:")
    print("  - datasets/train_aug/ (with augmentation)")
    print("  - datasets/valid_aug/ (copy only)")
    print("  - datasets/test_aug/ (copy only)")
    
    # Process train (with augmentation)
    train_total, train_ann = process_split('train', should_augment=True)
    
    # Process valid (no augmentation)
    valid_total, valid_ann = process_split('valid', should_augment=False)
    
    # Process test (no augmentation)
    test_total, test_ann = process_split('test', should_augment=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 ALL SPLITS PROCESSED!")
    print("=" * 80)
    print(f"\n📊 Final Summary:")
    print(f"   Train: {train_total} images, {train_ann} annotations")
    print(f"   Valid: {valid_total} images, {valid_ann} annotations")
    print(f"   Test: {test_total} images, {test_ann} annotations")
    print(f"\n✅ Now update train_resnet.py and train_faster_rcnn.py to use:")
    print(f"   - datasets/train_aug/_annotations.csv (instead of datasets/train)")
    print(f"   - datasets/valid_aug/_annotations.csv (instead of datasets/valid)")
    print(f"   - datasets/test_aug/_annotations.csv (instead of datasets/test)")
    print(f"\n🎯 This matches YOLO's augmented data for fair comparison!")


if __name__ == "__main__":
    main()
