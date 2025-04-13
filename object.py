import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the high-accuracy YOLOv8 model
# model = YOLO('yolov8l.pt')  # yolov8x.pt for even more accuracy
model = YOLO('yolov8s.pt')  # Or 'yolov8m.pt' if you still want decent accuracy


# Set model thresholds
model.conf = 0.6
model.iou = 0.45

# Allowed COCO class indices
allowed_classes = [0, 2, 3, 5, 7]  # Person, Car, Motorcycle, Bus, Truck

# Load video
video_path = 'test2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    exit()

frame_count = 0
fps = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Resize for YOLO
    resized = cv2.resize(frame, (640, 640))

    results = model(resized, verbose=False)[0]

    # Scale detections back to original resolution
    h_orig, w_orig = frame.shape[:2]
    scale_x = w_orig / 640
    scale_y = h_orig / 640

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id not in allowed_classes:
            continue

        if conf < 0.6:
            continue

        # Ignore tiny or huge detections
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < 1500 or box_area > 0.9 * (640 * 640):
            continue

        # Optionally ignore near-edge detections
        margin = 10
        if x1 < margin or y1 < margin or x2 > 640 - margin or y2 > 640 - margin:
            continue

        # Scale box to original frame
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        label = f"{model.names[cls_id]} {conf:.2f}"
        color = (0, 255, 0)

        # Optional debug print for suspicious detections
        if cls_id == 2 and conf < 0.75:
            print(f"🟡 Low-confidence car detected: {label} at ({x1},{y1},{x2},{y2})")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS
    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = end_time
        frame_count = 0

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('YOLOv8 High-Accuracy Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
