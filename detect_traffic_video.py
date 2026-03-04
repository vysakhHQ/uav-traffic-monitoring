from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture("C:/Users/mvysa/OneDrive/Desktop/Hangzhou Inshallah/traffic-monitoring-uav/traffic.mp4")
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

vehicle_classes = ["car", "bus", "truck", "motorcycle"]

while True:
    ret, frame = cap.read()

    if not ret:
        break
    print("Frame running") 
    results = model(frame, conf=0.25, imgsz=960)
    vehicle_count = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            label = model.names[cls]

            if label in vehicle_classes:
                vehicle_count += 1

        frame = r.plot()
    # Traffic density estimation
    if vehicle_count < 3:
        density = "LOW"
    elif vehicle_count < 7:
        density = "MEDIUM"
    else:
        density = "HIGH"
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )
    cv2.putText(
        frame,
        f"Traffic Density: {density}",
        (20,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )
    frame = cv2.resize(frame,(900,600))

    cv2.imshow("Traffic Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()