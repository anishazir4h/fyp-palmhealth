# src/check_yolo_pairs.py
from pathlib import Path

# set your YOLO dataset path
ROOT = Path("datasets/palms_yolo")

def check_split(split):
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"

    imgs = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]}
    lbls = {p.stem for p in lbl_dir.glob("*.txt")}

    missing_labels = imgs - lbls
    missing_images = lbls - imgs

    print(f"\n[{split.upper()}]")
    print(f" total images: {len(imgs)}")
    print(f" total labels: {len(lbls)}")

    if not missing_labels and not missing_images:
        print(" ✅ all images and labels match!")
    else:
        print(f" ⚠ missing label files: {len(missing_labels)}")
        print(f" ⚠ missing image files: {len(missing_images)}")

        if missing_labels:
            print("\n  Images without labels (first 10):")
            for n in list(missing_labels)[:10]:
                print("   ", n)
        if missing_images:
            print("\n  Labels without images (first 10):")
            for n in list(missing_images)[:10]:
                print("   ", n)

if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        check_split(split)
