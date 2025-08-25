import argparse, shutil, yaml, os
from pathlib import Path

def read_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(obj, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def slurp_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to IP102_YOLOv5 folder (contains images/, labels/, ip102.yaml)")
    ap.add_argument("--dst", required=True, help="Output subset folder, e.g., data/pest_subset")
    ap.add_argument("--classes", required=True,
                    help="Comma-separated class names to KEEP (matching YAML names exactly).")
    ap.add_argument("--splits", default="train,val", help="Comma-separated splits to process, default 'train,val'")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    names_keep = [c.strip() for c in args.classes.split(",") if c.strip()]

    # load source YAML (use ip102.yaml)
    yml_path = src / "ip102.yaml"
    yml = read_yaml(yml_path)

    # Build name->id map from source YAML
    src_names = yml["names"]
    if isinstance(src_names, list):
        id2name = {i: n for i, n in enumerate(src_names)}
        name2id = {n: i for i, n in id2name.items()}
    elif isinstance(src_names, dict):
        name2id = {v: int(k) for k, v in src_names.items()}
        id2name = {v: k for k, v in name2id.items()}
    else:
        raise ValueError("Unsupported names structure in YAML")

    # validate requested names
    missing = [n for n in names_keep if n not in name2id]
    if missing:
        raise SystemExit(f"❌ These class names were not found in source YAML: {missing}\n"
                         f"Tip: open {yml_path} and copy exact spellings.")

    keep_ids_ordered = [name2id[n] for n in names_keep]
    remap = {old_id: new_id for new_id, old_id in enumerate(keep_ids_ordered)}

    # Prepare output dirs
    splits = [s.strip() for s in args.splits.split(",")]
    for split in splits:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Walk each split and copy only images with at least one kept label
    kept_counts = {n: 0 for n in names_keep}
    total_kept_images = 0
    for split in splits:
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        out_img_dir = dst / "images" / split
        out_lbl_dir = dst / "labels" / split

        for img_path in sorted(img_dir.glob("*")):
            if img_path.is_dir():
                continue
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                continue

            lines = slurp_lines(lbl_path)
            out_lines = []
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id in remap:
                    new_id = remap[cls_id]
                    parts[0] = str(new_id)
                    out_lines.append(" ".join(parts))
                    # count by name
                    kept_counts[id2name[cls_id]] += 1

            # Keep image only if it has at least one kept label
            if out_lines:
                shutil.copy2(img_path, out_img_dir / img_path.name)
                with open(out_lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(out_lines) + "\n")
                total_kept_images += 1

    # Write subset YAML
    subset_yaml = {
        "path": str(dst),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names_keep),
        "names": names_keep,
    }
    write_yaml(subset_yaml, dst / "pest_subset.yaml")

    print("✅ Subset created at:", dst)
    print("   YAML:", dst / "pest_subset.yaml")
    print("   Total images kept:", total_kept_images)
    print("   Per-class kept box counts:")
    for n in names_keep:
        print(f"   - {n}: {kept_counts[n]}")

if __name__ == "__main__":
    main()
