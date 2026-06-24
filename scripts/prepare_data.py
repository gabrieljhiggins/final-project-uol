#!/usr/bin/env python3
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

def parse_voc_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
    except Exception as e:
        print(f"  [SKIP] Could not read {xml_path.name}: {e}")
        return 0, 0, []

    objects = []
    for obj in root.findall("object"):
        try:
            name = obj.find("name").text.strip().lower()
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            objects.append({
                "name": name,
                "x_center": round(max(0, min(1, x_center)), 6),
                "y_center": round(max(0, min(1, y_center)), 6),
                "width": round(max(0, min(1, bw)), 6),
                "height": round(max(0, min(1, bh)), 6),
            })
        except:
            continue
    return w, h, objects


def find_matching_image(xml_path):
    stem = xml_path.stem
    parent = xml_path.parent
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert XML annotations to YOLO format")
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Full path to your folder containing images and XML files")
    parser.add_argument("--output_dir", type=str, default="yolo_dataset")
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not data_root.exists():
        print(f"ERROR: Folder does not exist: {data_root}")
        return

    print(f"Scanning folder: {data_root}")

    # Create output folders
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    xml_files = list(data_root.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files")

    if len(xml_files) == 0:
        print("ERROR: No .xml files found. Check your path.")
        return

    # Collect all classes
    all_classes = set()
    for xml_file in xml_files:
        _, _, objs = parse_voc_xml(xml_file)
        for obj in objs:
            all_classes.add(obj["name"])

    class_list = sorted(list(all_classes))
    print(f"Classes found: {class_list}")
    class_to_id = {name: idx for idx, name in enumerate(class_list)}

    # Process files
    samples = []
    for xml_file in xml_files:
        w, h, objects = parse_voc_xml(xml_file)
        if w == 0 or len(objects) == 0:
            continue

        img_file = find_matching_image(xml_file)
        if img_file is None:
            print(f"  [WARN] No image found for {xml_file.name}")
            continue

        yolo_lines = []
        for obj in objects:
            cid = class_to_id[obj["name"]]
            line = f"{cid} {obj['x_center']} {obj['y_center']} {obj['width']} {obj['height']}"
            yolo_lines.append(line)

        samples.append({
            "stem": xml_file.stem,
            "img_path": img_file,
            "yolo_lines": yolo_lines
        })

    print(f"Valid samples ready: {len(samples)}")

    if len(samples) == 0:
        print("ERROR: No valid image + XML pairs found.")
        return

    # Split train / val
    random.seed(42)
    random.shuffle(samples)
    split = int(len(samples) * args.val_split)
    val_samples = samples[:split]
    train_samples = samples[split:]

    # Write files
    def write_data(items, img_folder, label_folder):
        for item in items:
            shutil.copy2(item["img_path"], img_folder / f"{item['stem']}{item['img_path'].suffix}")
            with open(label_folder / f"{item['stem']}.txt", "w") as f:
                f.write("\n".join(item["yolo_lines"]))

    write_data(train_samples, output_dir / "images/train", output_dir / "labels/train")
    write_data(val_samples, output_dir / "images/val", output_dir / "labels/val")

    # Create data.yaml
    data_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_list),
        "names": class_list
    }
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\nSUCCESS!")
    print(f"   YOLO dataset created at: {output_dir}")
    print(f"   Classes: {class_list}")
    print(f"   Train samples: {len(train_samples)}")
    print(f"   Val samples: {len(val_samples)}")


if __name__ == "__main__":
    main()