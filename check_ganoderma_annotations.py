import json

# Load the Ganoderma dataset
json_path = r"c:\Users\anish\OneDrive\Desktop\FYP\fyp-palm\datasets\ganoderma\Ganoderma Detection Dataset for Oil Palm Crop Disease Classification\train\_annotations.coco.json"

with open(json_path, 'r') as f:
    data = json.load(f)

print("=" * 60)
print("GANODERMA DATASET ANALYSIS")
print("=" * 60)

# Categories
print("\nCategories:")
for cat in data['categories']:
    print(f"  ID {cat['id']}: {cat['name']}")

# Check what's annotated
print(f"\nTotal images: {len(data['images'])}")
print(f"Total annotations: {len(data['annotations'])}")

# Check if all annotations are disease
all_category_ids = set(ann['category_id'] for ann in data['annotations'])
print(f"\nUnique category IDs used in annotations: {all_category_ids}")

# Sample image analysis
img = data['images'][0]
anns = [a for a in data['annotations'] if a['image_id'] == img['id']]

print(f"\n--- Sample Image Analysis ---")
print(f"Image: {img['file_name']}")
print(f"Size: {img['width']}x{img['height']}")
print(f"Number of bounding boxes: {len(anns)}")
print(f"All boxes are 'Ganoderma' (diseased): {all(a['category_id'] == 1 for a in anns)}")

# Count images with annotations
images_with_disease = len(set(ann['image_id'] for ann in data['annotations']))
images_without_annotations = len(data['images']) - images_with_disease

print(f"\n--- Annotation Distribution ---")
print(f"Images WITH Ganoderma disease detected: {images_with_disease}")
print(f"Images WITHOUT annotations (possibly healthy): {images_without_annotations}")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("✓ This dataset ONLY annotates UNHEALTHY trees (Ganoderma disease)")
print("✗ Healthy palms are NOT annotated (no bounding boxes)")
print("\nIf an image has no annotations, it likely contains:")
print("  - Only healthy palms (no disease visible)")
print("  - Or palms that weren't labeled")
print("=" * 60)
