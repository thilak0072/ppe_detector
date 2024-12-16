import cv2
from ultralytics import YOLO
path = './model.pt'
try:
    model = YOLO(path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
cap = cv2.VideoCapture('video2.mp4')
if not cap.isOpened():
    print("Error: Could not open video file!")
    exit()
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break
    count += 1
    if count % 3 != 0: 
        continue
    frame = cv2.resize(frame, (1020, 600))  
    try:
        results = model(frame)
    except Exception as e:
        print(f"Error during model inference: {e}")
        break
    try:
        annotated_frame = results[0].plot()
    except Exception as e:
        print(f"Error annotating frame: {e}")
        annotated_frame = frame
    cv2.imshow("FRAME", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC pressed, exiting.")
        break
cap.release()
cv2.destroyAllWindows()
