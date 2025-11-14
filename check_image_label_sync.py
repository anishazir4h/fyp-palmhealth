# Image and Label synchronizations
# Checks mismatches between images and labels in YOLO datasets

import os
from pathlib import Path
import shutil

def check_image_label_sync(dataset_path):
    """Check whether images and labels are properly matched"""
    
    image_dir = f"{dataset_path}/images"
    label_dir = f"{dataset_path}/labels"
    
    print(f"Checking dataset: {dataset_path}")
    print("=" * 60)
    
    image_files = set()
    label_files = set()
    
    # Get image filenames
    if os.path.exists(image_dir):
        for file in os.listdir(image_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                basename = os.path.splitext(file)[0]
                image_files.add(basename)
    
    # Get label filenames
    if os.path.exists(label_dir):
        for file in os.listdir(label_dir):
            if file.endswith('.txt'):
                basename = os.path.splitext(file)[0]
                label_files.add(basename)
    
    # Find mismatches
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files
    
    # Display summary
    print("Synchronization report:")
    print(f"   Images: {len(image_files)}")
    print(f"   Labels: {len(label_files)}")
    print(f"   Matched pairs: {len(image_files & label_files)}")
    print()
    
    if images_without_labels:
        print(f"Images without labels ({len(images_without_labels)}):")
        for i, filename in enumerate(sorted(images_without_labels)):
            if i < 10:
                print(f"   - {filename}")
            elif i == 10:
                print(f"   - ... and {len(images_without_labels) - 10} more")
        print()
    
    if labels_without_images:
        print(f"Labels without images ({len(labels_without_images)}):")
        for i, filename in enumerate(sorted(labels_without_images)):
            if i < 10:
                print(f"   - {filename}")
            elif i == 10:
                print(f"   - ... and {len(labels_without_images) - 10} more")
        print()
    
    if not images_without_labels and not labels_without_images:
        print("All images and labels are synchronized.")
        return True, [], []
    
    return False, list(images_without_labels), list(labels_without_images)

def fix_sync_issues(dataset_path, images_without_labels, labels_without_images, method="remove_unmatched"):
    """Fix synchronization problems between images and labels"""
    
    image_dir = f"{dataset_path}/images"
    label_dir = f"{dataset_path}/labels"
    
    print(f"\nFixing synchronization issues using method: {method}")
    print("=" * 60)
    
    if method == "remove_unmatched":
        removed_images = 0
        for filename in images_without_labels:
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(image_dir, f"{filename}{ext}")
                if os.path.exists(img_path):
                    os.remove(img_path)
                    removed_images += 1
                    break
        
        removed_labels = 0
        for filename in labels_without_images:
            label_path = os.path.join(label_dir, f"{filename}.txt")
            if os.path.exists(label_path):
                os.remove(label_path)
                removed_labels += 1
        
        print(f"Removed {removed_images} images without labels.")
        print(f"Removed {removed_labels} labels without images.")
    
    elif method == "create_dummy_labels":
        created_labels = 0
        for filename in images_without_labels:
            label_path = os.path.join(label_dir, f"{filename}.txt")
            with open(label_path, 'w') as f:
                f.write("")  # Empty file means no objects in image
            created_labels += 1
        
        print(f"Created {created_labels} empty label files.")
        print("Note: Empty labels mean 'no objects detected'. Review these cases manually if necessary.")

def main():
    """Check and fix synchronization for all dataset splits"""
    
    print("YOLO Dataset Image-Label Synchronization Tool")
    print("=" * 60)
    
    splits = ['train', 'train_aug', 'valid', 'test']
    
    for split in splits:
        dataset_path = f"datasets/palms_yolo/{split}"
        if os.path.exists(dataset_path):
            is_synced, images_no_labels, labels_no_images = check_image_label_sync(dataset_path)
            
            if not is_synced:
                print(f"Options to fix {split}:")
                print("   1. remove_unmatched: Remove files without pairs (recommended)")
                print("   2. create_dummy_labels: Create empty labels for unmatched images")
                
                if split == 'train_aug':
                    print(f"Automatically fixing {split} by removing unmatched files...")
                    fix_sync_issues(dataset_path, images_no_labels, labels_no_images, "remove_unmatched")
                
                print()
        else:
            print(f"Directory not found: {dataset_path}")
            print()
    
    # Final verification
    print("\nFinal verification:")
    print("=" * 60)
    for split in splits:
        dataset_path = f"datasets/palms_yolo/{split}"
        if os.path.exists(dataset_path):
            is_synced, _, _ = check_image_label_sync(dataset_path)
            status = "Synchronized" if is_synced else "Issues remaining"
            print(f"{split:12}: {status}")

if __name__ == "__main__":
    main()