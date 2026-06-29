#!/usr/bin/env python3
"""
Extract frames from your demo videos (Cam01.mp4, Cam02.mp4, Cam03.mp4)
All videos start at 2026-06-26 15:00:00
Saves into clean folders: Cam1/, Cam2/, Cam3/
Naming: Cam1_150000.jpg, Cam1_150001.jpg, ...
Also creates metadata.json in each folder.
"""

import cv2
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse
from tqdm import tqdm

def extract_cam(video_path: Path, cam_name: str, output_root: Path, start_dt: datetime, fps: float):
    if not video_path.exists():
        print(f"[SKIP] Video not found: {video_path}")
        return

    cam_dir = output_root / cam_name
    cam_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing {cam_name} from {video_path.name} ({total_frames} frames)...")

    start_ts = start_dt.timestamp()
    frame_interval = 1.0 / fps
    meta = []

    for frame_idx in tqdm(range(total_frames), desc=cam_name, unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        ts = start_ts + (frame_idx * frame_interval)
        wall_time = (start_dt + timedelta(seconds=frame_idx * frame_interval)).strftime("%H:%M:%S.%f")[:-3]

        
        numeric_part = 150000 + frame_idx
        filename = f"{cam_name}_{numeric_part:06d}.jpg"
        filepath = cam_dir / filename

        cv2.imwrite(str(filepath), frame)

        meta.append({
            "local_idx": frame_idx,
            "ts": round(ts, 3),
            "annotated_name": filename,
            "orig_name": video_path.name,
            "wall_time": wall_time
        })

    cap.release()

    with open(cam_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  → Saved {len(meta)} frames → {cam_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, 
                        default=Path("/Users/gabrielhiggins/Desktop/demo_videos"),
                        help="Folder containing Cam01.mp4, Cam02.mp4, Cam03.mp4")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_frames"))
    parser.add_argument("--start", type=str, default="2026-06-26 15:00:00")
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    args.output_dir.mkdir(exist_ok=True)

    print(f"Start time: {start_dt}")
    print(f"Reading videos from: {args.input_dir}\n")

    
    video_map = {
        1: ("Cam01.mp4", "Cam1"),
        2: ("Cam02.mp4", "Cam2"),
        3: ("Cam03.mp4", "Cam3"),
    }

    for cam_num, (video_filename, output_folder) in video_map.items():
        video_path = args.input_dir / video_filename
        extract_cam(video_path, output_folder, args.output_dir, start_dt, args.fps)

    print("\n Finished!")
    print(f"Frames saved in: {args.output_dir.resolve()}")
    print("Folders created: Cam1/, Cam2/, Cam3/")


if __name__ == "__main__":
    main()
