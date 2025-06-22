import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import numpy as np
import os
from datetime import datetime
from tracker import Tracker
import cvzone

# === Load models đã huấn luyện ===
vehicle_model = YOLO("yolov10s.pt")  # Model phát hiện phương tiện
model_light = YOLO(r"train14\weights\best.pt")  # Model phát hiện đèn

# === Load class list ===
with open("coco.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# === Video input và thư mục output ===
video_path = "tr.mp4"
cap = cv2.VideoCapture(video_path)
tracker = Tracker()
output_dir = os.path.join("saved_images", datetime.now().strftime("%Y-%m-%d"))
os.makedirs(output_dir, exist_ok=True)

# === Vùng chọn ROI
vehicle_points = []
direction_points = []
current_state = 'VEHICLE'

def unified_click_event(event, x, y, flags, param):
    global vehicle_points, direction_points, current_state, first_frame
    points_list = vehicle_points if current_state == 'VEHICLE' else direction_points
    color = (255, 0, 0) if current_state == 'VEHICLE' else (0, 255, 255)
    max_points = 4

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_list) >= max_points:
            points_list.pop(0)
        points_list.append((x, y))

        temp_frame = first_frame.copy()
        msg = f"Click to define {'traffic light' if current_state == 'VEHICLE' else 'detection'} area ({max_points - len(points_list)} points left)"
        cv2.putText(temp_frame, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        for i, point in enumerate(points_list):
            cv2.circle(temp_frame, point, 5, color, -1)
            if i > 0:
                cv2.line(temp_frame, points_list[i - 1], point, color, 2)
        if len(points_list) == max_points:
            cv2.line(temp_frame, points_list[-1], points_list[0], color, 2)
        cv2.imshow('Unified Frame', temp_frame)

def unified_select_roi(label, first_frame, points_list):
    global current_state
    current_state = label
    temp = first_frame.copy()
    msg = "Click to define traffic light area (4 points needed)" if label == 'VEHICLE' else "Click to define detection area (4 points needed)"
    cv2.putText(temp, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('Unified Frame', temp)
    cv2.setMouseCallback('Unified Frame', unified_click_event)
    cv2.waitKey(0)
    polygon = np.array(points_list, np.int32)
    return polygon

# === Đọc frame đầu để chọn vùng
ret, first_frame = cap.read()
if not ret:
    print("Không thể đọc video.")
    exit()
first_frame = cv2.resize(first_frame, (1020, 600))

cv2.namedWindow('Unified Frame')
print("Select traffic light area:")
traffic_light_polygon = unified_select_roi('VEHICLE', first_frame, vehicle_points)
print("Select detection area:")
detection_polygon = unified_select_roi('DIRECTION', first_frame, direction_points)
cv2.destroyAllWindows()

# === Vòng lặp chính
frame_count = 0
traffic_light_status = "UNKNOWN"
violation_ids = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 600))
    frame_count += 1

    # --- Detect vehicles
    vehicle_results = vehicle_model(frame)
    vehicle_detections = vehicle_results[0].boxes.data if vehicle_results else []

    object_list = []
    for det in vehicle_detections:
        x1, y1, x2, y2, _, cls_id = map(int, det.tolist())
        class_name = class_list[cls_id]
        object_list.append([x1, y1, x2, y2, class_name])

    tracked_objects = tracker.update([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in object_list])

    # --- Detect traffic light every 3 frames
    if frame_count % 3 == 0:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [traffic_light_polygon], 255)
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        results = model_light(frame)
        preds = results[0].boxes.data.cpu().numpy() if results else []
        labels = results[0].names if results else []

        classes = [labels[int(det[5])] for det in preds]
        if "red" in classes:
            traffic_light_status = "RED"
        elif "yellow" in classes:
            traffic_light_status = "YELLOW"
        elif "green" in classes:
            traffic_light_status = "GREEN"
        else:
            traffic_light_status = "UNKNOWN"

    # --- Check violation
    for tracked_obj, (_, _, _, _, class_name) in zip(tracked_objects, object_list):
        x, y, w, h, obj_id = tracked_obj
        cx, cy = (x + x + w) // 2, (y + y + h) // 2

        if cv2.pointPolygonTest(detection_polygon, (cx, cy), False) >= 0:
            if class_name == "car" and traffic_light_status == "RED":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'ID: {obj_id}', (x, y - 10), 1, 1, colorR=(0, 0, 255))
                if obj_id not in violation_ids:
                    violation_ids.append(obj_id)
                    img_path = os.path.join(output_dir, f"violation_{obj_id}.jpg")
                    cv2.imwrite(img_path, frame)
                    print(f"Saved violation: {img_path}")
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'ID: {obj_id}', (x, y - 10), 1, 1, colorR=(0, 255, 0))

    # --- Overlay
    cv2.putText(frame, f"Traffic Light: {traffic_light_status}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if traffic_light_status == "RED" else
                (0, 255, 255) if traffic_light_status == "YELLOW" else
                (0, 255, 0) if traffic_light_status == "GREEN" else (255, 255, 255), 2)

    cv2.polylines(frame, [traffic_light_polygon], True, (0, 255, 0), 2)
    cv2.polylines(frame, [detection_polygon], True, (255, 0, 0), 2)

    cv2.imshow("Traffic Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
