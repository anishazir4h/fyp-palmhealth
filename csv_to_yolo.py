# src/csv_to_yolo_per_split.py
from pathlib import Path
import pandas as pd, shutil, json

HERE = Path(__file__).resolve().parent
PROJ = HERE
SRC_ROOT = PROJ / "datasets"
DST_ROOT = PROJ / "datasets" / "palms_yolo"

def ensure_dir(p): p.mkdir(parents=True, exist_ok=True)

def convert_split(split):
    src_dir = SRC_ROOT / split
    csv_path = src_dir / "_annotations.csv"
    df = pd.read_csv(csv_path)

    img_dst = DST_ROOT / split / "images"
    lbl_dst = DST_ROOT / split / "labels"
    ensure_dir(img_dst); ensure_dir(lbl_dst)

    # build index of existing images (case-insensitive)
    disk_imgs = {p.name.lower(): p for p in src_dir.glob("*.jpg")}
    total, copied, written = 0, 0, 0

    # class mapping (sorted for consistent YOLO ids)
    classes = sorted(df["class"].unique())
    class_to_id = {c:i for i,c in enumerate(classes)}

    for img_name, group in df.groupby("filename"):
        total += 1
        base = Path(img_name).name.lower()
        src_img = disk_imgs.get(base)
        if src_img:
            shutil.copy2(src_img, img_dst / src_img.name)
            copied += 1

        lines = []
        for _, r in group.iterrows():
            W, H = r["width"], r["height"]
            x1, y1, x2, y2 = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
            # normalize to 0–1
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            cid = class_to_id[r["class"]]
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # write label file
        if lines:
            out = lbl_dst / (Path(img_name).stem + ".txt")
            out.write_text("\n".join(lines))
            written += 1

    print(f"[{split}] rows={len(df)} images_copied={copied}/{len(disk_imgs)} labels_written={written}")
    return classes

if __name__ == "__main__":
    ensure_dir(DST_ROOT)
    all_classes = set()
    for s in ["train", "valid", "test"]:
        cls = convert_split(s)
        all_classes |= set(cls)
    (DST_ROOT / "classes.json").write_text(json.dumps(sorted(all_classes), indent=2))
    print("✅ Done! YOLO dataset ready at:", DST_ROOT.resolve())
