import cv2
import time
from ultralytics import YOLO

# === Configuration ===
VIDEO_PATH = "test2.mp4"     
MODEL_PATH = "yolov8m.pt"         # Use 'yolov8l.pt' for higher accuracy (if GPU available)
CONFIDENCE_THRESHOLD = 0.75       # 🔐 Filter out low confidence predictions
MAX_ASPECT_RATIO = 3.5            # 🛑 Prevent wide + flat (false) boxes
MIN_HEIGHT = 30                   # 📏 Skip very short boxes (often road markings)

# === Load YOLOv8 model ===
model = YOLO(MODEL_PATH)

# === Open video ===
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"❌ Error opening video: {VIDEO_PATH}"

# === Output video writer ===
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output_filtered.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# === Frame-by-frame detection loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # === Run inference ===
    results = model(frame, conf=CONFIDENCE_THRESHOLD)
    boxes = results[0].boxes

    # === Annotate high-quality detections only ===
    annotated = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.model.names[cls_id]

        width_box = x2 - x1
        height_box = y2 - y1
        aspect_ratio = width_box / (height_box + 1e-6)

        # === Filter based on shape to ignore false road detections ===
        if height_box < MIN_HEIGHT or aspect_ratio > MAX_ASPECT_RATIO:
            continue  # skip sketchy box

        # === Draw bounding box & label ===
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 128, 255), 2)
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 128, 255), 2)

    # === Show FPS ===
    fps_val = 1 / (time.time() - start_time)
    cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 128, 255), 3)


    # === Display & save ===
    cv2.imshow("ADAS Smart Detection", annotated)
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Clean up ===
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Video saved as 'output_filtered1.mp4'")
