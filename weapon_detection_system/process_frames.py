#!/usr/bin/env python3
"""
Multi-Camera Weapon Detection Processor
- Processes pre-extracted frames from multiple camera segments
- Segments have known real-world start times (for chronological sync)
- Runs YOLO weapon detection + tracking
- Outputs annotated frames + metadata.json per camera + global timeline.json
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# ==================== COMPLETE START TIMES (from your data) ====================
SEGMENT_STARTS = {
    # Cam1
    ("Cam1", "Segment_0"): datetime(2018, 9, 22, 9, 26, 53),
    ("Cam1", "Segment_1"): datetime(2018, 9, 22, 9, 29, 29),
    ("Cam1", "Segment_2"): datetime(2018, 9, 22, 9, 34, 1),
    ("Cam1", "Segment_3"): datetime(2018, 9, 22, 9, 40, 50),
    ("Cam1", "Segment_4"): datetime(2018, 9, 22, 9, 49, 17),
    ("Cam1", "Segment_5"): datetime(2018, 9, 22, 9, 59, 56),

    # Cam5
    ("Cam5", "Segment_0"): datetime(2018, 9, 22, 9, 24, 19),
    ("Cam5", "Segment_1"): datetime(2018, 9, 22, 9, 25, 19),
    ("Cam5", "From09-27-50"): datetime(2018, 9, 22, 9, 27, 51),
    ("Cam5", "From09-30-04"): datetime(2018, 9, 22, 9, 30, 5),
    ("Cam5", "From09-32-26"): datetime(2018, 9, 22, 9, 32, 27),
    ("Cam5", "From09-46-24"): datetime(2018, 9, 22, 9, 46, 25),
    ("Cam5", "From09-47-49"): datetime(2018, 9, 22, 9, 47, 50),

    # Cam7
    ("Cam7", "Segment_6"):  datetime(2018, 9, 22, 9, 24, 5),
    ("Cam7", "Segment_7"):  datetime(2018, 9, 22, 9, 24, 27),
    ("Cam7", "Segment_8"):  datetime(2018, 9, 22, 9, 26, 12),
    ("Cam7", "Segment_9"):  datetime(2018, 9, 22, 9, 27, 32),
    ("Cam7", "Segment_10"): datetime(2018, 9, 22, 9, 29, 10),
    ("Cam7", "Segment_11"): datetime(2018, 9, 22, 9, 29, 50),
    ("Cam7", "Segment_13"): datetime(2018, 9, 22, 9, 39, 27),
}


def get_camera_and_key(path: Path):
    name = path.name
    cam_match = re.search(r'(Cam\d+)', name, re.IGNORECASE)
    cam = cam_match.group(1).capitalize() if cam_match else "Unknown"

    seg_match = re.search(r'Segment_(\d+)', name)
    if seg_match:
        return cam, f"Segment_{seg_match.group(1)}"

    from_match = re.search(r'From(\d{2}-\d{2}-\d{2})', name)
    if from_match:
        return cam, f"From{from_match.group(1)}"

    return cam, "unknown"


def get_frame_number(path: Path) -> int:
    m = re.search(r'frame_(\d+)', path.name)
    return int(m.group(1)) if m else 0


def main():
    p = argparse.ArgumentParser(description="Process multi-cam frames for weapon detection + timeline")
    p.add_argument("--model", required=True, help="Path to YOLO .pt model (weapon detection)")
    p.add_argument("--image_dir", type=Path, required=True, help="Directory containing all frame_*.jpg (mixed cams/segments)")
    p.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Where to write annotated/ and timeline.json")
    args = p.parse_args()

    model = YOLO(str(args.model))
    out = args.output_dir.resolve()
    out.mkdir(exist_ok=True, parents=True)
    ann_dir = out / "annotated"
    ann_dir.mkdir(exist_ok=True, parents=True)

    # Anything that is not 'person'/'human' is considered a weapon (customize if needed)
    weapons = {cid: name for cid, name in model.names.items()
               if str(name).lower() not in ["person", "human"]}

    all_imgs = [x for x in args.image_dir.iterdir()
                if x.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    cams_dict = defaultdict(list)
    for img in all_imgs:
        cam, _ = get_camera_and_key(img)
        cams_dict[cam].append(img)

    events = []

    for cam, img_list in cams_dict.items():
        print(f"\nProcessing {cam} ({len(img_list)} frames)...")

        groups = defaultdict(list)
        for img in img_list:
            _, key = get_camera_and_key(img)
            groups[key].append(img)

        for key in groups:
            groups[key].sort(key=get_frame_number)

        # === FIX: Sort segments by REAL start time (not lexical key order!) ===
        segment_info = []
        for key in groups:
            start_dt = SEGMENT_STARTS.get((cam, key), datetime(2018, 9, 22, 9, 0, 0))
            segment_info.append((start_dt, key))
        segment_info.sort(key=lambda x: x[0])  # chronological by actual capture start

        sorted_frames = []
        for start_dt, key in segment_info:
            sorted_frames.extend(groups[key])

        cam_ann_dir = ann_dir / cam
        cam_ann_dir.mkdir(exist_ok=True, parents=True)

        frame_meta = []
        last_log_ts = -999.0

        for local_idx, img_path in enumerate(tqdm(sorted_frames, desc=cam)):
            cam_name, key = get_camera_and_key(img_path)

            start_dt = SEGMENT_STARTS.get((cam_name, key), datetime(2018, 9, 22, 9, 0, 0))
            if (cam_name, key) not in SEGMENT_STARTS:
                print(f"  [WARNING] No start time found for {img_path.name} (key={key})")

            frame_num = get_frame_number(img_path)
            event_dt = start_dt + timedelta(seconds=frame_num * 0.5)
            ts = event_dt.timestamp()

            res = model.track(str(img_path), persist=True, verbose=False)[0]

            weaps = []
            for b in (res.boxes or []):
                cid = int(b.cls[0])
                if cid in weapons:
                    weaps.append({"name": weapons[cid], "conf": float(b.conf[0])})

            if weaps and (ts - last_log_ts > 1.5):
                names = sorted(set(w["name"] for w in weaps))
                events.append({
                    "time": event_dt.strftime("%H:%M:%S"),
                    "seconds": round(ts, 2),
                    "camera": cam,
                    "event": f"Weapon visible: {', '.join(names)}",
                    "frame_index": local_idx,
                    "image_name": img_path.name
                })
                last_log_ts = ts

            out_name = f"frame_{local_idx:05d}.jpg"
            try:
                cv2.imwrite(str(cam_ann_dir / out_name), res.plot())
            except Exception:
                pass

            frame_meta.append({
                "local_idx": local_idx,
                "ts": ts,
                "annotated_name": out_name,
                "orig_name": img_path.name,
                "wall_time": event_dt.strftime("%H:%M:%S")  # helpful for display
            })

        with open(cam_ann_dir / "metadata.json", "w") as f:
            json.dump(frame_meta, f, indent=2)

    events.sort(key=lambda e: e["seconds"])
    with open(out / "timeline.json", "w") as f:
        json.dump(events, f, indent=2)

    print(f"\nDONE! Total weapon events logged: {len(events)}")
    print(f"Annotated frames + metadata in: {ann_dir}")
    print(f"Timeline: {out / 'timeline.json'}")


if __name__ == "__main__":
    main()
