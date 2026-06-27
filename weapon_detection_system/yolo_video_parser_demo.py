#!/usr/bin/env python3
"""
Extract frames from your demo videos at target FPS (e.g. 2 FPS).
Videos: video1.mp4, video2.mp4, video3.mp4
Saves into: Cam1/, Cam2/, Cam3/
"""

import cv2
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse
from tqdm import tqdm

def extract_cam(video_path: Path, cam_name: str, output_root: Path, start_dt: datetime, target_fps: float):
    if not video_path.exists():
        print(f"[SKIP] Video not found: {video_path}")
        return

    cam_dir = output_root / cam_name
    cam_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing {cam_name} from {video_path.name} ({total_frames} frames, orig FPS: {orig_fps:.2f})...")

    # Calculate frame interval for target FPS
    frame_interval = 1.0 / target_fps
    start_ts = start_dt.timestamp()

    meta = []
    saved_count = 0

    for frame_idx in tqdm(range(total_frames), desc=cam_name, unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        # Only save frames at the target rate
        if frame_idx % max(1, int(orig_fps / target_fps)) == 0:
            ts = start_ts + (frame_idx * (1.0 / orig_fps))  # real timestamp from original video
            wall_time = (start_dt + timedelta(seconds=frame_idx / orig_fps)).strftime("%H:%M:%S.%f")[:-3]

            numeric_part = 150000 + saved_count
            filename = f"{cam_name}_{numeric_part:06d}.jpg"
            filepath = cam_dir / filename

            cv2.imwrite(str(filepath), frame)

            meta.append({
                "local_idx": saved_count,
                "orig_frame_idx": frame_idx,
                "ts": round(ts, 3),
                "annotated_name": filename,
                "orig_name": video_path.name,
                "wall_time": wall_time
            })
            saved_count += 1

    cap.release()

    with open(cam_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  → Saved {saved_count} frames at ~{target_fps} FPS → {cam_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, 
                        default=Path("/Users/gabrielhiggins/Desktop/yolo_train_videos"),
                        help="Folder containing video1.mp4, video2.mp4, video3.mp4")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_train_frames_2fps"), help="Output folder")
    parser.add_argument("--start", type=str, default="2026-06-26 15:00:00")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS (default: 2)")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    args.output_dir.mkdir(exist_ok=True)

    print(f"Start time: {start_dt}")
    print(f"Target FPS: {args.fps}")
    print(f"Reading videos from: {args.input_dir}\n")

    video_map = {
        1: ("video1.mp4", "Cam1"),
        2: ("video2.mp4", "Cam2"),
        3: ("video3.mp4", "Cam3"),
    }

    for cam_num, (video_filename, output_folder) in video_map.items():
        video_path = args.input_dir / video_filename
        extract_cam(video_path, output_folder, args.output_dir, start_dt, args.fps)

    print("\n✅ Finished!")
    print(f"Frames saved in: {args.output_dir.resolve()}")
    print("Folders created: Cam1/, Cam2/, Cam3/")


if __name__ == "__main__":
    main()