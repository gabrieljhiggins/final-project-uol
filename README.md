# Real-Time Weapon Detection and Event Reconstruction in Multi-Camera Surveillance Systems
# Final Project UoL - Preliminary Prototype: Weapon Detection with YOLOv26 + Basic Timeline Recreation

## Overview
This preliminary prototype demonstrates the core components of the proposed real-time surveillance pipeline:
- **Simple weapon (and person) detection** using a **YOLOv26** model (Ultralytics latest, edge-optimized with native end-to-end NMS-free inference) trained/fine-tuned on your custom multi-weapon dataset (Pascal VOC XML annotations from `~/Desktop/Images`).
- **Basic timeline recreation** using the public **Mock Attack Dataset** (real CCTV frames from staged university attack, 3 cameras, annotated at 2 FPS). The script processes frames sequentially (simulating video stream), runs inference, logs weapon detections over "time", performs simple spatial association of weapons to persons, and generates timestamped event timelines (JSON + text summary + matplotlib visualization).

This fulfills the Planning & Evaluation phase requirement for a 3-5 min video demo of early system functionality (motivation + core integration). It aligns with the hybrid edge-cloud architecture: training in cloud (here simulated), inference runnable on edge (RPi + Hailo ready via export).

**Note on YOLOv26**: Released 2025/2026 by Ultralytics. Use `yolo26n.pt` (nano) or `yolo26s.pt` for edge efficiency. It features NMS-free inference, ProgLoss, STAL for small objects (critical for distant CCTV weapons), MuSGD optimizer.

**Dataset Notes**:
- **Custom**: Your labeled images + XML in `~/Desktop/Images` (assumed flat dir with paired `image.jpg` + `image.xml` from labelImg or similar; supports .jpg/.png/.jpeg). Script auto-discovers classes and converts to YOLO format.
- **Mock Attack**: ~5k frames from Cam1/Cam5/Cam7. Download automatically or manually from HF. Used **unlabeled for inference** to demo detection + timeline on realistic CCTV attack sequence. (GT available in original but format not standardized in zip; prototype focuses on model predictions for end-to-end flow).

**Full Pipeline Context** (from project):
- Perception: YOLOv26 joint person + multi-weapon detection.
- Association: Spatial (IoU/proximity) link weapon -> tracked person (simple version here; full uses Re-ID + tracking).
- Timeline: Event logs for alarms, forensic review. (Here basic per-camera event JSON; full syncs to cloud dashboard).

Later phases will add: Hailo export/optimization, ByteTrack/SORT + Hailo Re-ID for persistent tracking, cross-cam stitching, full threat association, cloud archiving, dashboard (Streamlit/FastAPI).

## Quick Start (Prototype Demo)

### 1. Setup Environment
```bash
cd final-project-uol-prototype
python -m venv venv
source venv/bin/activate  # or conda
pip install -r requirements.txt
# For YOLO26 pre-trained weights auto-download on first use
```

(Recommended: Use GPU machine or Colab for training. For edge test later: export to ONNX/TensorRT then Hailo compiler.)

### 2. Prepare Custom Dataset (from your Desktop/Images)
```bash
python scripts/prepare_custom_dataset.py
```
- Scans `~/Desktop/Images` (configurable via `--images_dir`).
- Discovers unique classes from XML `<name>` tags (e.g. person, handgun, knife, rifle...).
- Converts VOC XML -> YOLO .txt labels (normalized xywh).
- Splits 80/20 train/val (stratified-ish by image count).
- Creates `data/custom_yolov26.yaml` with paths and class names.
- Output: `data/custom/` with images/train|val/ and labels/train|val/ + data yaml.

**Edit** `data/custom_yolov26.yaml` if needed (e.g. add more classes or change paths). Re-run if you add images.

### 3. Train YOLOv26 Model
```bash
python scripts/train_yolov26.py --data data/custom_yolov26.yaml --model yolo26n.pt --epochs 50 --imgsz 640 --batch 16
```
- Fine-tunes from COCO pre-trained YOLOv26 nano (fast, edge-friendly; ~3-6M params).
- Saves best.pt to `models/yolov26_weapon_person_best.pt` (and last.pt, results plots).
- Uses augmentations good for CCTV (mosaic, mixup, etc. via Ultralytics).
- Logs to `runs/detect/trainX/`.
- **Tip**: For preliminary, 20-50 epochs sufficient to show concept. Monitor mAP50-95, especially for small weapons.
- After training, update `BEST_MODEL_PATH` in other scripts if needed.

**Expected**: Good detection on your labeled styles; transfer to mock CCTV may need fine-tune later (domain gap: lighting, angle, resolution).

### 4. Run Inference + Basic Timeline Recreation on Mock Attack Dataset
```bash
python scripts/inference_timeline_mock.py --model models/yolov26_weapon_person_best.pt --output_dir timelines/mock_attack_demo
```
- Auto-downloads `weapons_images_2fps.zip` (~few hundred MB) from HF to `data/mock/` if not present (or use `--mock_dir`).
- Extracts if needed.
- For each camera (Cam1, Cam5, Cam7): 
  - Globs & sorts frames (assumes lexical order = temporal; 2 FPS).
  - Runs YOLOv26 inference (conf=0.25, iou=0.45).
  - **Simple association**: For each weapon detection, finds nearest person (center Euclidean dist) if persons present; logs "threat associated to person at [bbox]".
  - Builds per-frame events + aggregated incidents (weapon visible periods).
- Outputs to `timelines/mock_attack_demo/`:
  - `timeline_events.json`: Full list of timestamped detections + associations (ready for dashboard ingest).
  - `timeline_summary.txt`: Human-readable event log per camera (e.g. "Cam7: Weapon (handgun conf=0.82) first detected at ~t=45s (frame 90), associated to person... Duration of threat: 12s").
  - `timeline_plot.png`: Matplotlib timeline showing detection events over "video time" (color-coded by camera/class).
  - `sample_annotated_frames/`: Few example frames with drawn bboxes, labels, assoc lines (for video demo screenshots).
  - Optional: `--save_video` to create simple MP4 per cam (annotated, slow but useful for demo; requires ffmpeg or uses cv2).
- **Simulates real pipeline**: Processes "stream" sequentially, low-latency per frame ready for edge.

Run time: ~minutes on GPU for all ~5k frames (or subsample with `--max_frames 200` for quick test).

### 5. Demo Video (3-5 min) Suggestions
Record screen while:
1. Showing project motivation slides (surveillance challenges, hybrid edge-cloud, bias mitigation via association).
2. Running prepare + short train (or load pre-trained if time).
3. Running timeline script, show live console logs of detections, open JSON/txt/plot.
4. Highlight: YOLOv26 NMS-free good for edge, simple assoc as stepping stone to full Re-ID tracking, timeline for forensic/alarms.
5. Mention next: Hailo deploy, full tracking, cloud dashboard, real multi-cam tests.

## File Structure
```
final-project-uol-prototype/
├── README.md
├── requirements.txt
├── data/
│   ├── custom_yolov26.yaml          # Generated by prepare
│   ├── custom/                      # YOLO formatted custom data (train/val)
│   └── mock/                        # Mock attack images (downloaded)
├── scripts/
│   ├── prepare_custom_dataset.py
│   ├── train_yolov26.py
│   └── inference_timeline_mock.py
├── models/
│   └── yolov26_weapon_person_best.pt  # After training
├── timelines/
│   └── mock_attack_demo/            # Generated timelines, plots, samples
├── docs/
│   └── architecture_notes.md        # (extend with your report excerpts)
└── runs/                            # Ultralytics training logs (auto)
```

## Customization & Extension Notes (for your full project)
- **Classes**: Script auto-detects from your XML. Common for this domain: person, handgun, knife, rifle, shotgun, etc. Update mapping if weapon subtypes.
- **Association (simple -> advanced)**: Current = nearest person by bbox center. Full impl: use IoU + motion + Re-ID embedding (Hailo Re-ID model) + tracker (ByteTrack) to maintain persistent person IDs across frames/occlusions. Weapon linked to track ID.
- **Timeline**: JSON schema extensible for cloud sync (add event_type: 'weapon_detected', 'threat_confirmed', severity, track_id). Full version correlates across cams via Re-ID gallery for backtracking.
- **Edge Optimization**: After train, `model.export(format='onnx')` or TensorRT. Then use Hailo Dataflow Compiler + SDK for RPi5 + Hailo-8. YOLO26's end-to-end design simplifies this (no NMS post-proc).
- **Mock Dataset Citation**: Always cite Salazar González et al. (2020) when using/publishing timelines or results from it. Academic only (CC BY-NC 4.0).
- **Your Custom Data**: Ensure XML valid Pascal VOC (from labelImg, CVAT, etc.). Add more images/annotations for better mAP, especially varied CCTV conditions.
- **Limitations of Proto**: No real tracking/Re-ID (single frame assoc), no cross-cam, training on desktop not full cloud, mock uses images not live video. No privacy (faces blurred? add later). No alarms UI.
- **Next Steps for Report/Video**:
  - Quantify proto perf: mAP on val split, inference FPS on CPU/GPU/Jetson.
  - Qualitative: Show failure cases (occlusion, small objects) -> motivate full pipeline.
  - Hardware: Mention RPi5 + Hailo setup done in parallel (procurement weeks 9-10).
  - Ethical: Bias mitigation via explicit person-weapon association (avoids "weapon present = alert anyone").

## Troubleshooting
- XML parse error: Check your annotations have valid <bndbox> with xmin/xmax/ymin/ymax numeric. Some tools use different tags.
- No detections on mock: Domain shift (your labels vs real CCTV). Solution: Add mock images to custom train (semi-sup or pseudo-label), or lower conf thresh.
- Training slow/OOM: Use smaller imgsz=416, batch=8, or yolo26n, or --workers 2. Or use Ultralytics hub/Colab.
- Frame order wrong: Inspect `data/mock/weapons_images_2fps/CamX/` filenames; if not sortable, add timestamp parsing (prototype assumes padded numbers or lexical time order).
- YOLOv26 not found: Ensure latest ultralytics (`pip install -U ultralytics`). Weights download from GitHub/Utralytics on first `YOLO("yolo26n.pt")`.

## References
- Project Concept & Architecture (your report sections).
- Salazar González, J.L. et al. (2020). Real-time gun detection in CCTV: An open problem. Neural Networks.
- Ultralytics YOLO26 Docs: https://docs.ultralytics.com/models/yolo26 (NMS-free, edge focus perfect for IoT/Hailo).
- Dataset: https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset (cite paper).

This prototype provides a solid, runnable foundation to demonstrate technical integration for your preliminary report and video. Extend it iteratively toward the full edge-deployed system with tracking and Re-ID.
