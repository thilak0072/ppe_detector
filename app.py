import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="PPE Detection Demo", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: black;
            color: white;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: white;
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            margin: 0;
            background-color: black;
            padding: 10px 0;
            z-index: 9999;
        }
        .center-text {
            text-align: center;
        }
    </style>
    <div class="footer">2024 @Copyright by Embrais AI Solutions. All Rights Reserved.</div>
""", unsafe_allow_html=True)
st.title('PPE Detection Demo')
st.progress(100)

@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        model.fuse()
        model.info(verbose=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

path = './model.pt'
model = load_model(path)
if model is None:
    st.stop()

video_options = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
selected_video = st.selectbox("Select a video for PPE detection:", video_options)
ppe_options = ["All", "Hardhat", "Safety Vest", "Mask", "Person"]
selected_ppe = st.radio("Select PPE to detect:", ppe_options)

if selected_video:
    st.subheader(f"Selected video: {selected_video}")
    video_path = f'./{selected_video}'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file!")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown('<div class="center-text">Processed frames with detections will be displayed here:</div>', unsafe_allow_html=True)
        processed_frame_placeholder = st.empty()
        progress_bar = st.progress(0)

    conf = 0.4
    iou =0.4

    frame_count = 0
    skip_frames = 5
    detection_classes = {
        "Hardhat": ["Hardhat", "NO-Hardhat"],
        "Safety Vest": ["Safety Vest", "NO-Safety Vest"],
        "Mask": ["Mask", "NO-Mask"],
        "Person": ["Person"]
    }

    while True:
        ret, frame = cap.read()

        if not ret:
            st.info("End of video.")
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        frame = cv2.resize(frame, (640, 480))

        try:
            results = model(frame, conf=0.4, iou=0.4, verbose=False, device='cpu', half=False, max_det=10)
            annotated_frame = frame.copy()

            for result in results[0].boxes:
                bbox = result.xyxy[0].cpu().numpy()
                cls = int(result.cls[0].item())
                conf = result.conf[0].item()
                label = results[0].names[cls]
                color = None
                text = None

                if selected_ppe == "All" or label in detection_classes.get(selected_ppe, []):
                    if label in ["Hardhat", "Safety Vest", "Mask", "Person"]:
                        color = (0, 255, 0)
                        text = label
                    elif label in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]:
                        color = (0, 0, 255)
                        text = label

                if color and text:
                    cv2.rectangle(annotated_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.putText(annotated_frame, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        except Exception as e:
            st.error(f"Inference error: {e}")
            annotated_frame = frame

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        processed_frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

        progress = min(int((frame_count / total_frames) * 100), 100)
        progress_bar.progress(progress)

    cap.release()
    progress_bar.progress(0)
