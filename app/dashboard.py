import streamlit as st
import cv2
import json
import pandas as pd
from pathlib import Path
import tempfile
import os
from datetime import datetime

st.set_page_config(page_title="Weapon Detection Dashboard", layout="wide")
st.title("🔫 Multi-Camera Weapon Detection Dashboard")

# ====================== CONFIG ======================
ANNOTATED_FRAMES_DIR = Path("outputs/annotated")   # Change if needed
TIMELINE_JSON = Path("outputs/timeline.json")

# ====================== LOAD DATA ======================
@st.cache_data
def load_timeline():
    if TIMELINE_JSON.exists():
        with open(TIMELINE_JSON) as f:
            return json.load(f)
    return []

timeline_events = load_timeline()

# ====================== SIDEBAR ======================
st.sidebar.header("Controls")
camera_names = ["Cam1", "Cam5", "Cam7"]
selected_cameras = st.sidebar.multiselect("Select Cameras", camera_names, default=camera_names)

# ====================== LIVE CAMERA VIEWS ======================
st.header("📹 Live Camera Feeds")

cols = st.columns(len(selected_cameras))

for idx, cam in enumerate(selected_cameras):
    with cols[idx]:
        st.subheader(f"Camera: {cam}")
        
        # Get frames for this camera (you can improve filtering later)
        frames = sorted(list(ANNOTATED_FRAMES_DIR.glob("*.jpg")))
        
        if frames:
            # Simple "live" simulation - cycle through frames
            frame_index = st.session_state.get(f"{cam}_frame", 0)
            current_frame = frames[frame_index % len(frames)]
            
            st.image(str(current_frame), use_column_width=True, caption=f"Frame {frame_index}")
            
            # Auto-advance simulation
            if st.button(f"▶ Next Frame ({cam})", key=f"next_{cam}"):
                st.session_state[f"{cam}_frame"] = frame_index + 1
                st.rerun()
        else:
            st.warning(f"No annotated frames found for {cam}")

# ====================== TIMELINE ======================
st.header("Event Timeline")

if timeline_events:
    df = pd.DataFrame(timeline_events)
    st.dataframe(df, use_container_width=True)
    
    # Simple timeline visualization
    st.subheader("Weapon Detections Over Time")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        st.line_chart(df.set_index("time")["track"].value_counts().sort_index())
else:
    st.info("No timeline events found. Run the timeline_recreation.py script first.")

# ====================== DOWNLOAD WEAPON VIDEO ======================
st.header("Download Weapon Frames Video")

if st.button("Generate & Download Weapon Video"):
    weapon_frames = sorted(list(ANNOTATED_FRAMES_DIR.glob("*.jpg")))
    
    if not weapon_frames:
        st.error("No annotated frames found in outputs/annotated/")
    else:
        with st.spinner("Creating video from weapon frames..."):
            # Create video
            first_frame = cv2.imread(str(weapon_frames[0]))
            h, w, _ = first_frame.shape
            
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_path = temp_video.name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 5, (w, h))  # 5 FPS
            
            for frame_path in weapon_frames:
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    out.write(frame)
            
            out.release()
            
            # Offer download
            with open(video_path, "rb") as f:
                st.download_button(
                    label="Download Weapon Frames Video",
                    data=f,
                    file_name=f"weapon_frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4"
                )
            
            st.success(f"Video created with {len(weapon_frames)} frames!")

st.caption("Note: This is a preliminary dashboard for your project demo.")