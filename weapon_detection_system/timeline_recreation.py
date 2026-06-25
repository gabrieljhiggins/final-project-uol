#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

def get_camera(name):
    import re
    m = re.search(r'(Cam\d+)', name, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unknown"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image_dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    model = YOLO(str(args.model))
    out = args.output_dir.resolve()
    out.mkdir(exist_ok=True)
    ann_dir = out / "annotated"
    ann_dir.mkdir(exist_ok=True)

    # All non-person classes = weapons
    weapons = {}
    for cid, name in model.names.items():
        if str(name).lower() not in ["person", "human"]:
            weapons[cid] = str(name)

    print(f"Weapon classes: {list(weapons.values())}")

    all_imgs = sorted([x for x in args.image_dir.iterdir()
                       if x.suffix.lower() in {".jpg",".jpeg",".png"}])

    cams_dict = defaultdict(list)
    for img in all_imgs:
        cams_dict[get_camera(img.name)].append(img)

    events = []
    interval = 1.0 / args.fps

    for cam, img_list in cams_dict.items():
        print(f"Processing {cam} ({len(img_list)} frames)...")
        t = 0.0
        cam_ann_dir = ann_dir / cam
        cam_ann_dir.mkdir(exist_ok=True)
        last_log = -999

        for idx, img_path in enumerate(tqdm(img_list, desc=cam)):
            res = model.track(str(img_path), persist=True, verbose=False)[0]

            weaps = []
            for b in (res.boxes or []):
                cid = int(b.cls[0])
                if cid in weapons:
                    weaps.append({"name": weapons[cid], "conf": float(b.conf[0])})

            if weaps and (t - last_log > 1.5):
                names = sorted(set(w["name"] for w in weaps))
                events.append({
                    "time": str(timedelta(seconds=int(t))),
                    "seconds": round(t, 2),
                    "camera": cam,
                    "event": f"Weapon visible: {', '.join(names)}",
                    "weapons": names,
                    "conf": round(max(w["conf"] for w in weaps), 3),
                    "frame_index": idx,
                    "image_name": img_path.name
                })
                last_log = t

            try:
                cv2.imwrite(str(cam_ann_dir / f"frame_{idx:05d}.jpg"), res.plot())
            except:
                pass
            t += interval

    events.sort(key=lambda e: (e["camera"], e["seconds"]))
    with open(out / "timeline.json", "w") as f:
        json.dump(events, f, indent=2)

    print(f"\nDone! Events: {len(events)}")
    print(f"Annotated frames → outputs/annotated/<Cam>/")
    print(f"Timeline → outputs/timeline.json")

if __name__ == "__main__":
    main()