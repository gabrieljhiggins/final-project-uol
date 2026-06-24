#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0: return 0
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image_dir", type=Path, required=True)
    p.add_argument("--max_frames", type=int, default=80)
    p.add_argument("--camera", default="Cam1")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    model = YOLO(args.model)
    out = args.output_dir
    out.mkdir(exist_ok=True)
    ann_dir = out / "annotated"
    ann_dir.mkdir(exist_ok=True)

    person_id = None
    weapons = {}
    for cid, name in model.names.items():
        if str(name).lower() == "person":
            person_id = cid
        else:
            weapons[cid] = str(name)

    imgs = sorted([x for x in args.image_dir.iterdir() if x.suffix.lower() in [".jpg",".png",".jpeg"]])
    if args.max_frames:
        imgs = imgs[:args.max_frames]

    events = []
    history = defaultdict(lambda: {"weapons": set()})
    t = 0.0
    interval = 1.0 / args.fps

    for idx, img in enumerate(tqdm(imgs)):
        res = model.track(str(img), persist=True, verbose=False)[0]
        boxes = res.boxes or []

        persons, weaps = [], []
        for b in boxes:
            cid = int(b.cls[0])
            conf = float(b.conf[0])
            xy = b.xyxy[0].tolist()
            tid = int(b.id[0]) if b.id is not None else -1
            if person_id is not None and cid == person_id:
                persons.append({"tid": tid, "box": xy, "conf": conf})
            elif cid in weapons:
                weaps.append({"name": weapons[cid], "box": xy, "conf": conf})

        if person_id is not None and persons and weaps:
            for w in weaps:
                best = max(persons, key=lambda p: iou(w["box"], p["box"]))
                if iou(w["box"], best["box"]) > 0.1 and best["tid"] != -1:
                    if w["name"] not in history[best["tid"]]["weapons"]:
                        history[best["tid"]]["weapons"].add(w["name"])
                        events.append({
                            "time": str(timedelta(seconds=int(t))),
                            "track": best["tid"],
                            "event": f"acquired {w['name']}",
                            "conf": round(w["conf"], 3)
                        })

        try:
            cv2.imwrite(str(ann_dir / f"frame_{idx:05d}.jpg"), res.plot())
        except:
            pass
        t += interval

    with open(out / "timeline.json", "w") as f:
        json.dump(events, f, indent=2)

    print("\n=== TIMELINE ===")
    for e in events[:30]:
        print(f"[{e['time']}] Track#{e['track']}: {e['event']}")
    print(f"\nSaved to {out / 'timeline.json'}")

if __name__ == "__main__":
    main()