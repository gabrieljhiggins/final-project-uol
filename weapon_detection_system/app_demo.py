#!/usr/bin/env python3
"""
Demo Dashboard for Weapon Detection
Run with: python app_demo.py --port 8080
"""

from flask import Flask, render_template, send_file, jsonify, request
from pathlib import Path
import json
import cv2
import numpy as np
import tempfile
import zipfile
from io import BytesIO

from camera_utils import (
    get_cam_data,
    get_available_cams,
    get_global_time_range,
    get_closest_frame,
    load_timeline as _load_timeline,
)

# ==================== CONFIG ====================
BASE_DIR = Path(__file__).parent

# Default: look for outputs/ one level up (final-project-uol/outputs)
DEFAULT_OUTPUT = BASE_DIR.parent / "outputs"

ANNOTATED_DIR = DEFAULT_OUTPUT / "annotated"
TIMELINE_JSON = DEFAULT_OUTPUT / "timeline.json"

app = Flask(__name__, template_folder="templates")


def load_timeline():
    return _load_timeline(TIMELINE_JSON)


# ==================== ROUTES ====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/frames/<cam>/<int:idx>")
def serve_frame(cam, idx):
    data = get_cam_data(cam, ANNOTATED_DIR)
    if not data["frames"] or idx < 0 or idx >= len(data["frames"]):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "NO FEED", (180, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (180, 180, 180), 3)
        cv2.putText(img, cam, (220, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 120, 120), 2)
        _, buf = cv2.imencode(".jpg", img)
        return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")
    return send_file(str(data["frames"][idx]["path"]))


@app.route("/api/cams")
def api_cams():
    # Demo camera order: Cam1 → Cam2 → Cam3
    all_cams = get_available_cams(ANNOTATED_DIR)
    preferred_order = ["Cam1", "Cam2", "Cam3"]
    cams = [c for c in preferred_order if c in all_cams] + \
           [c for c in all_cams if c not in preferred_order]

    info = {}
    for cam in cams:
        d = get_cam_data(cam, ANNOTATED_DIR)
        info[cam] = {
            "frame_count": len(d["frames"]),
            "min_ts": d["min_ts"],
            "max_ts": d["max_ts"]
        }
    gmin, gmax = get_global_time_range(ANNOTATED_DIR, TIMELINE_JSON)
    return jsonify({
        "cams": cams,
        "info": info,
        "global_min_ts": gmin,
        "global_max_ts": gmax
    })


@app.route("/api/timeline")
def api_timeline():
    return jsonify(load_timeline())


@app.route("/api/closest/<cam>")
def api_closest(cam):
    target = float(request.args.get("time", 0))
    frame = get_closest_frame(cam, target, ANNOTATED_DIR)
    if not frame:
        return jsonify({"idx": -1})
    return jsonify({
        "idx": frame["local_idx"],
        "ts": frame["ts"],
        "diff": abs(frame["ts"] - target),
        "wall_time": frame.get("wall_time", "")
    })


@app.route("/api/download_archive", methods=["POST"])
def download_archive():
    events = load_timeline()
    tmp = tempfile.mkdtemp()
    zpath = Path(tmp) / "events_archive.zip"
    with zipfile.ZipFile(zpath, 'w') as zf:
        if TIMELINE_JSON.exists():
            zf.write(TIMELINE_JSON, "timeline.json")
        for ev in events[:30]:
            cam = ev.get("camera")
            fidx = int(ev.get("frame_index", 0))
            data = get_cam_data(cam, ANNOTATED_DIR)
            if data["frames"] and 0 <= fidx < len(data["frames"]):
                zf.write(data["frames"][fidx]["path"], f"{cam}/event_{ev.get('time', 'unknown')}.jpg")
    return send_file(zpath, as_attachment=True, download_name="weapon_events_archive.zip")


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Demo Dashboard - Weapon Detection")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Path to outputs folder (must contain annotated/ and timeline.json)")
    args = p.parse_args()

    if args.output_dir:
        ANNOTATED_DIR = args.output_dir / "annotated"
        TIMELINE_JSON = args.output_dir / "timeline.json"
        print(f"[INFO] Using custom output dir: {args.output_dir}")

    print(f"Dashboard running → http://127.0.0.1:{args.port}")
    print(f"Looking for annotated frames in: {ANNOTATED_DIR}")
    app.run(host="0.0.0.0", port=args.port, debug=True)