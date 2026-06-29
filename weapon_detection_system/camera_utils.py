#!/usr/bin/env python3
"""
Camera data loading and time-sync utilities.
"""

from pathlib import Path
import json
import re
from bisect import bisect_left
from typing import Dict, List, Any, Optional


def parse_timestamp_from_filename(path: Path) -> float:
    name = path.name
    match = re.search(r'From(\d{2})-(\d{2})-(\d{2})', name)
    if match:
        h, m, s = map(int, match.groups())
        base = h * 3600 + m * 60 + s
        fm = re.search(r'frame_(\d+)', name)
        if fm:
            base += int(fm.group(1)) * 0.5
        return base
    return 0.0


def get_cam_data(cam: str, annotated_dir: Path) -> Dict[str, Any]:
    p = annotated_dir / cam
    if not p.exists():
        return {"frames": [], "timestamps": [], "min_ts": 0, "max_ts": 0}

    meta_file = p / "metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                meta: List[Dict] = json.load(f)
            data = []
            for m in meta:
                local_idx = int(m.get("local_idx", 0))
                ann_name = m.get("annotated_name", f"frame_{local_idx:05d}.jpg")
                fpath = p / ann_name
                if fpath.exists():
                    data.append({
                        "path": fpath,
                        "ts": float(m.get("ts", 0.0)),
                        "local_idx": local_idx,
                        "wall_time": m.get("wall_time", "")
                    })
            data.sort(key=lambda x: x["ts"])
            if data:
                return {
                    "frames": data,
                    "timestamps": [d["ts"] for d in data],
                    "min_ts": data[0]["ts"],
                    "max_ts": data[-1]["ts"]
                }
        except Exception as e:
            print(f"[WARN] metadata.json error for {cam}: {e}")

    # Legacy fallback
    frames = sorted(p.glob("frame_*.jpg"))
    data = []
    for local_idx, fpath in enumerate(frames):
        ts = parse_timestamp_from_filename(fpath)
        data.append({"path": fpath, "ts": ts, "local_idx": local_idx, "wall_time": ""})
    data.sort(key=lambda x: x["ts"])

    if data:
        return {
            "frames": data,
            "timestamps": [d["ts"] for d in data],
            "min_ts": data[0]["ts"],
            "max_ts": data[-1]["ts"]
        }
    return {"frames": [], "timestamps": [], "min_ts": 0, "max_ts": 0}


def get_available_cams(annotated_dir: Path) -> List[str]:
    if not annotated_dir.exists():
        return []
    return sorted([d.name for d in annotated_dir.iterdir()
                   if d.is_dir() and any(d.glob("frame_*.jpg"))])


def get_global_time_range(annotated_dir: Path, timeline_json: Path) -> tuple:
    all_ts = []
    for cam in get_available_cams(annotated_dir):
        d = get_cam_data(cam, annotated_dir)
        if d["timestamps"]:
            all_ts.extend([d["min_ts"], d["max_ts"]])
    if timeline_json.exists():
        try:
            with open(timeline_json) as f:
                for ev in json.load(f):
                    if isinstance(ev, dict) and "seconds" in ev:
                        all_ts.append(float(ev["seconds"]))
        except Exception:
            pass
    return (min(all_ts), max(all_ts)) if all_ts else (0.0, 0.0)


def get_closest_frame(cam: str, target_ts: float, annotated_dir: Path) -> Optional[Dict]:
    data = get_cam_data(cam, annotated_dir)
    if not data["timestamps"]:
        return None

    ts_list = data["timestamps"]
    pos = bisect_left(ts_list, target_ts)

    candidates = []
    if pos < len(ts_list):
        candidates.append(data["frames"][pos])
    if pos > 0:
        candidates.append(data["frames"][pos - 1])

    if not candidates:
        return data["frames"][0]

    return min(candidates, key=lambda f: abs(f["ts"] - target_ts))


def load_timeline(timeline_json: Path) -> list:
    if not timeline_json.exists():
        return []
    try:
        with open(timeline_json) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def seconds_to_hms(seconds: float) -> str:
    from datetime import timedelta
    return str(timedelta(seconds=int(seconds)))[-8:]
