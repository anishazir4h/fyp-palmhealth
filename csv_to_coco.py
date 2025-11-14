# src/csv_to_coco_per_split.py
from pathlib import Path
import pandas as pd
import json, shutil
from datetime import datetime

HERE = Path(__file__).resolve().parent
PROJ = HERE
SRC_ROOT = PROJ / "datasets"          # expects datasets/train, datasets/valid, datasets/test
DST_ROOT = PROJ / "datasets" / "palms_coco"

IMG_EXTS = (".jpg", ".jpeg", ".png")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_images(src_dir: Path):
    """Index images on disk (case-insensitive) by filename."""
    disk = {}
    for ext in IMG_EXTS:
        for p in src_dir.glob(f"*{ext}"):
            disk[p.name.lower()] = p
    return disk

def clip_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(float(x1), W - 1))
    y1 = max(0, min(float(y1), H - 1))
    x2 = max(0, min(float(x2), W - 1))
    y2 = max(0, min(float(y2), H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return x1, y1, w, h

def collect_all_classes(splits=("train","valid","test")):
    all_classes = set()
    for s in splits:
        csv_path = SRC_ROOT / s / "_annotations.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "class" in df.columns:
                all_classes |= set(df["class"].dropna().astype(str).tolist())
    classes = sorted(all_classes)
    # COCO category ids are usually 1..N
    class_to_id = {c: i+1 for i, c in enumerate(classes)}
    return classes, class_to_id

def convert_split(split: str, class_to_id: dict):
    src_dir = SRC_ROOT / split
    csv_path = src_dir / "_annotations.csv"
    if not csv_path.exists():
        print(f"[{split}] ❌ Missing CSV: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    # normalize types
    for col in ["width","height","xmin","ymin","xmax","ymax"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build destination dirs
    img_dst = DST_ROOT / "images" / split
    ann_dst = DST_ROOT / "annotations"
    ensure_dir(img_dst); ensure_dir(ann_dst)

    # Disk image index
    disk_imgs = find_images(src_dir)

    # COCO containers
    coco = {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": f"Palms COCO ({split}) converted from CSV",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": name, "supercategory": "palm"} 
                       for name, cid in class_to_id.items()]
    }

    # Image id & annotation id counters
    image_id = 1
    ann_id = 1

    # Track mapping filename -> image_id to avoid duplicates within a split
    filename_to_imgid = {}

    # Sort by filename for reproducibility
    df = df.copy()
    df["filename_lc"] = df["filename"].astype(str).str.lower()
    df = df.sort_values(["filename_lc"]).reset_index(drop=True)

    copied_count = 0
    written_anns = 0
    missing_imgs = 0

    # Loop per image
    for fname, g in df.groupby("filename_lc"):
        # Locate image on disk
        src_img = disk_imgs.get(fname)
        if src_img is None:
            missing_imgs += 1
            continue

        # Get width/height: prefer provided columns; fallback to first row
        first = g.iloc[0]
        W = int(first["width"]) if pd.notna(first["width"]) else None
        H = int(first["height"]) if pd.notna(first["height"]) else None

        # If width/height missing, try reading via PIL (optional) — skip to keep dependencies minimal
        if W is None or H is None:
            # You can uncomment this if you want auto-read sizes:
            # from PIL import Image
            # with Image.open(src_img) as im:
            #     W, H = im.size
            raise ValueError(f"Width/Height missing for {src_img.name}. Please include in CSV or enable PIL size read.")

        # Copy image
        dst_img_path = img_dst / src_img.name
        if not dst_img_path.exists():
            shutil.copy2(src_img, dst_img_path)
            copied_count += 1

        # Register image in COCO
        coco["images"].append({
            "id": image_id,
            "file_name": src_img.name,
            "width": W,
            "height": H
        })
        filename_to_imgid[fname] = image_id

        # Add annotations for this image
        for _, r in g.iterrows():
            cls_name = str(r["class"])
            if cls_name not in class_to_id:
                # Skip unknown class (shouldn't happen if we pre-collected classes)
                continue
            cid = int(class_to_id[cls_name])

            x1, y1, x2, y2 = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
            if pd.isna([x1, y1, x2, y2]).any():
                continue

            # Clip, convert to COCO [x,y,w,h] in absolute pixels
            x, y, w, h = clip_bbox(x1, y1, x2, y2, W, H)
            if w <= 0 or h <= 0:
                continue

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cid,
                "bbox": [round(float(x), 3), round(float(y), 3), round(float(w), 3), round(float(h), 3)],
                "area": round(float(w * h), 3),
                "iscrowd": 0,
                "segmentation": []  # not using masks
            })
            ann_id += 1
            written_anns += 1

        image_id += 1

    # Save JSON
    out_json = ann_dst / f"instances_{split}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[{split}] images={len(coco['images'])} copied={copied_count}  annos={written_anns}  missing_imgs={missing_imgs}  -> {out_json}")
    return coco

if __name__ == "__main__":
    ensure_dir(DST_ROOT / "images" / "train")
    ensure_dir(DST_ROOT / "images" / "valid")
    ensure_dir(DST_ROOT / "images" / "test")
    ensure_dir(DST_ROOT / "annotations")

    # Make categories consistent across splits
    classes, class_to_id = collect_all_classes()
    (DST_ROOT / "categories.json").write_text(json.dumps({
        "classes": classes,
        "class_to_id": class_to_id
    }, indent=2))

    for split in ["train", "valid", "test"]:
        convert_split(split, class_to_id)

    print("✅ Done! COCO dataset ready at:", DST_ROOT.resolve())
