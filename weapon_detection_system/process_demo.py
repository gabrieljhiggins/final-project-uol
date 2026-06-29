#!/usr/bin/env python3
"""
Clean Demo Processor - Weapon Detection
Saves frames as frame_XXXXX.jpg for compatibility
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("../outputs"))
    args = parser.parse_args()

    model = YOLO(str(args.model))
    out_dir = args.output_dir.resolve()
    ann_dir = out_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    weapons = {cid: name for cid, name in model.names.items()
               if str(name).lower() not in ["person", "human"]}

    events = []
    cameras = ["Cam1", "Cam2", "Cam3"]

    for cam in cameras:
        cam_input_dir = args.image_dir / cam
        meta_file = cam_input_dir / "metadata.json"

        if not meta_file.exists():
            print(f"[WARNING] Skipping {cam} - no metadata.json")
            continue

        with open(meta_file) as f:
            frame_meta = json.load(f)

        print(f"Processing {cam} ({len(frame_meta)} frames)...")

        cam_ann_dir = ann_dir / cam
        cam_ann_dir.mkdir(exist_ok=True)

        new_meta = []
        last_log_ts = -999.0

        for item in tqdm(frame_meta, desc=cam):
            img_path = cam_input_dir / item["annotated_name"]
            if not img_path.exists():
                continue

            frame_idx = item["local_idx"]
            ts = item["ts"]

            results = model.track(str(img_path), persist=True, verbose=False)[0]

            # Weapon detection
            weaps = []
            for box in (results.boxes or []):
                cid = int(box.cls[0])
                if cid in weapons:
                    weaps.append(weapons[cid])

            if weaps and (ts - last_log_ts > 1.5):
                names = sorted(set(weaps))
                events.append({
                    "time": item["wall_time"],
                    "seconds": round(ts, 2),
                    "camera": cam,
                    "event": f"Weapon visible: {', '.join(names)}",
                    "frame_index": frame_idx,
                    "image_name": f"frame_{frame_idx:05d}.jpg"
                })
                last_log_ts = ts

            # Save as standard .jpg
            out_name = f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(cam_ann_dir / out_name), results.plot())

            new_meta.append({
                "local_idx": frame_idx,
                "ts": ts,
                "annotated_name": out_name,
                "wall_time": item["wall_time"]
            })

        with open(cam_ann_dir / "metadata.json", "w") as f:
            json.dump(new_meta, f, indent=2)

    events.sort(key=lambda e: e["seconds"])
    with open(out_dir / "timeline.json", "w") as f:
        json.dump(events, f, indent=2)

    print(f"\n Done! Total weapon events: {len(events)}")
    print(f"Annotated frames saved in: {ann_dir}")


if __name__ == "__main__":
    main()
