import os
import sys
import torch
import cv2
import math
import warnings
import numpy as np
from collections import deque

warnings.filterwarnings("ignore", category=FutureWarning)

def get_dominant_color(image, k=1):
    """Get dominant color from image using k-means clustering"""
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    dominant_color = centers[0].astype(int)
    return tuple(dominant_color)

def color_similarity(c1, c2, threshold=50):
    """Check if two colors are similar within threshold"""
    return np.sqrt(sum((x-y)**2 for x,y in zip(c1,c2))) < threshold

def run_detection(video_path="api/uploads/test_theft.mp4", output_dir="api/outputs"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found!")


    print("[INFO] Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.25  # General confidence threshold
    model.classes = [0, 1, 3]  # Detect people(0), bicycles(1), motorcycles(3)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file '{video_path}'")

    frame_size = (int(cap.get(3)), int(cap.get(4)))
    output_path = os.path.join(output_dir, "scooter_monitoring.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_size)

    state = {
        'parked': False,
        'owner_color': None,
        'owner_position': None,
        'stable_frames': 0,
        'last_position': None,
        'theft_alert': False,
        'parking_threshold': 15,  # 0.5 seconds at 30 FPS
        'movement_threshold': 35.0,
        'min_person_area': 5000  # Minimum area for valid person detection
    }

    frame_count = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        
        current_scooter = None
        people = []
        
        for *box, conf, cls in detections:
            class_id = int(cls)
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            
            if class_id in [1, 3]:  # Scooter detection
                if not current_scooter or area > current_scooter['area']:
                    current_scooter = {
                        'box': (x1, y1, x2, y2),
                        'center': ((x1+x2)//2, (y1+y2)//2),
                        'area': area
                    }
            elif class_id == 0 and area > state['min_person_area']:  # Person detection
                people.append({
                    'box': (x1, y1, x2, y2),
                    'lower_body': (x1, min(y2, y1 + int((y2-y1)*0.6)), x2, y2)
                })
                # Draw person bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 100, 50), 2)


        if current_scooter:
            # Movement detection
            if state['last_position']:
                dx = current_scooter['center'][0] - state['last_position'][0]
                dy = current_scooter['center'][1] - state['last_position'][1]
                distance = math.hypot(dx, dy)
                
                if distance < state['movement_threshold']:
                    state['stable_frames'] += 1
                else:
                    state['stable_frames'] = 0
                    state['parked'] = False

            # Parking logic
            if not state['parked'] and state['stable_frames'] >= state['parking_threshold']:
                state['parked'] = True
                state['owner_color'] = None
                state['owner_position'] = None
                
                if people:
                    closest = min(people, key=lambda p: math.hypot(
                        (p['box'][0]+p['box'][2])//2 - current_scooter['center'][0],
                        (p['box'][1]+p['box'][3])//2 - current_scooter['center'][1]
                    ))
                    x1, y1, x2, y2 = closest['lower_body']
                    pants_roi = frame[y1:y2, x1:x2]
                    if pants_roi.size > 0:
                        state['owner_color'] = get_dominant_color(pants_roi)
                        state['owner_position'] = current_scooter['center']
                        # Draw owner marker
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Owner", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            state['last_position'] = current_scooter['center']
            
            # Draw scooter box
            x1, y1, x2, y2 = current_scooter['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 2)
            cv2.putText(frame, "Scooter", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            state['stable_frames'] = 0
            state['parked'] = False

        # -----------------------------
        # 8) Enhanced Theft Detection
        # -----------------------------
        state['theft_alert'] = False
        if state['parked'] and current_scooter:
            # Alert if scooter moves without owner
            if state['last_position'] and state['owner_position']:
                movement_distance = math.hypot(
                    state['last_position'][0] - state['owner_position'][0],
                    state['last_position'][1] - state['owner_position'][1]
                )
                if movement_distance > state['movement_threshold']:
                    state['theft_alert'] = True

            # Check for unauthorized users
            if people and state['owner_color']:
                for person in people:
                    x1, y1, x2, y2 = person['lower_body']
                    pants_roi = frame[y1:y2, x1:x2]
                    if pants_roi.size == 0:
                        continue
                    
                    current_color = get_dominant_color(pants_roi)
                    if not color_similarity(current_color, state['owner_color']):
                        state['theft_alert'] = True
                        # Draw intruder alert
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, "INTRUDER!", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        break

        status_text = "Monitoring..."
        status_color = (200, 150, 0)  # Orange
        
        if state['parked']:
            status_text = "Scooter Parked"
            status_color = (0, 200, 0)  # Green
            if state['theft_alert']:
                status_text = "THEFT DETECTED!"
                status_color = (0, 0, 255)  # Red
                cv2.putText(frame, "ALERT: Unauthorized Movement!", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Draw status box
        cv2.rectangle(frame, (20, 20), (350, 80), status_color, -1)
        cv2.putText(frame, status_text, (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Draw detection counters
        cv2.putText(frame, f"People: {len(people)}", (frame.shape[1]-200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[SUCCESS] Output saved to: {output_path}")

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "api/uploads/test_theft.mp4"
    run_detection(video_path=video_path)
