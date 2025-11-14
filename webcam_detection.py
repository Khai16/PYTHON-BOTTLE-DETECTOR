from ultralytics import YOLO
import cv2

# Load YOLOv8 small model (downloads automatically on first run)
model = YOLO("yolov8n.pt")

# COCO bottle class index
BOTTLE_CLASS_ID = 39

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detect only bottle class
    results = model.predict(frame, conf=0.5, classes=[BOTTLE_CLASS_ID], verbose=False)

    # Draw boxes for bottles
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Bottle {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Bottle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("e"):
        break

cap.release()
cv2.destroyAllWindows()
