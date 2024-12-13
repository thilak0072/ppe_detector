import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO

# Set page configuration
st.set_page_config(page_title="PPE Detection Demo", layout="wide")

# Styling remains the same
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

# Title
st.title('PPE Detection Demo')
st.progress(100)

# Model loading with performance optimization
@st.cache_resource
def load_model(path):
    try:
        # Use optimal settings for faster inference
        model = YOLO(path)
        model.fuse()  # Fuse model layers for faster inference
        
        # Additional performance optimizations
        model.info(verbose=False)  # Disable verbose logging
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Path to the model
path = './model.pt'
model = load_model(path)

if model is None:
    st.stop()

# Video selection
video_options = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
selected_video = st.selectbox("Select a video for PPE detection:", video_options)

if selected_video:  
    st.subheader(f"Selected video: {selected_video}")
    video_path = f'./{selected_video}'
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: Could not open video file!")
        st.stop()
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create columns for layout
    col1, col2, col3 = st.columns([1, 4, 1])  
    
    with col2:
        st.markdown('<div class="center-text">Processed frames with detections will be displayed here:</div>', unsafe_allow_html=True)
        processed_frame_placeholder = st.empty()
        progress_bar = st.progress(0)
    
    # Inference configuration for speed
    conf = 0.4  # Lower confidence threshold for faster processing
    iou = 0.4   # Lower IOU threshold for faster processing
    
    # Frame processing loop with speed optimizations
    frame_count = 0
    skip_frames = 3  # Process every frame (adjust to 2 or 3 for even more speed)
    
    while True:
        ret, frame = cap.read()
        
        # End of video
        if not ret:
            st.info("End of video.")
            break
        
        # Skip frames for performance
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Perform inference with optimized settings
        try:
            # Run inference with specified confidence and IOU thresholds
            results = model(
                frame, 
                conf=conf, 
                iou=iou, 
                verbose=False,  # Disable verbose output
                device='cpu',  # Force CPU (change to '0' for GPU if available)
                half=False,    # Disable half precision for CPU
                max_det=10     # Limit max detections to speed up processing
            )
            
            # Plot results with minimal annotations
            annotated_frame = results[0].plot(
                conf=True,   # Show confidence scores
                labels=True, # Show class labels
                boxes=True   # Show bounding boxes
            )
        except Exception as e:
            st.error(f"Inference error: {e}")
            annotated_frame = frame
        
        # Convert to RGB for Streamlit display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display the processed frame
        processed_frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
        
        # Update progress
        progress = min(int((frame_count / total_frames) * 100), 100)
        progress_bar.progress(progress)
        
        # Minimal delay to improve performance
        st.empty()  # Help with Streamlit performance
    
    # Release resources
    cap.release()
    
    # Reset progress
    progress_bar.progress(0)