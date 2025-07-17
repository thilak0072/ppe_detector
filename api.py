from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from typing import List

app = FastAPI()

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("Dricz/ppe-obj-detection")
model = AutoModelForObjectDetection.from_pretrained("Dricz/ppe-obj-detection")

@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(('.mp4', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload")

    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Open the video
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail="Failed to process video. Please check the file.")

        frame_results = []  # To store detection results for each frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit when video ends

            # Preprocess the frame
            inputs = processor(images=frame, return_tensors="pt")

            # Perform inference
            outputs = model(**inputs)

            # Decode results
            target_sizes = [frame.shape[:2]]
            results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

            # Store detection results
            frame_result = []
            for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
                frame_result.append({
                    "label": model.config.id2label[label.item()],
                    "score": score.item(),
                    "box": box.tolist(),
                })
            frame_results.append(frame_result)

        cap.release()
        os.remove(temp_file_path)  # Cleanup the temporary file

        # Return detection results
        return JSONResponse(content={"detections": frame_results}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if __name__=="__main__":
        uvicorn.run(app, host="127.0.0.1", port=8000)
