import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# Initialize YOLO model for person and scooter detection
model = YOLO('yolov8n.pt')  # Replace with custom-trained model if needed
# model = YOLO('escooter_detector_model.h5')  # Replace with custom-trained model if needed

# Dictionary to store scooter and associated user features
scooter_registry = defaultdict(dict)

# Helper function to extract dominant color from a bounding box
def extract_dominant_color(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image[y1:y2, x1:x2]
    avg_color = cv2.mean(cropped)[:3]
    return tuple(map(int, avg_color))

# Function to check if two colors are similar
def is_color_similar(color1, color2, threshold=50):
    return sum(abs(c1 - c2) for c1, c2 in zip(color1, color2)) < threshold

def draw_status_bar(frame, message, color=(0, 0, 255)):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), color, -1)
    cv2.putText(frame, message, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# Main function to process video feed
def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        results = model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        current_scooters = []
        current_people = []

        # Process detections
        for bbox, cls, conf in zip(detections, classes, confidences):
            if conf < 0.5:
                continue
            
            if int(cls) == 3 or int(cls) == 1:  # Scooter class
                current_scooters.append(bbox)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, "Scooter", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
            
            elif int(cls) == 0:  # Person class
                current_people.append(bbox)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(frame, "Person", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)

        # Associate people with scooters
        for scooter_bbox in current_scooters:
            for person_bbox in current_people:
                if is_nearby(scooter_bbox, person_bbox): 
                    dominant_color = extract_dominant_color(frame, person_bbox)
                    scooter_registry[tuple(scooter_bbox)]["features"] = dominant_color
                    scooter_registry[tuple(scooter_bbox)]["timestamp"] = time.time()
                    break

        # Detect suspicious activity
        alert_triggered = False
        for scooter_bbox in current_scooters:
            scooter_key = tuple(scooter_bbox)
            if scooter_key in scooter_registry:
                registered_features = scooter_registry[scooter_key]["features"]
                for person_bbox in current_people:
                    dominant_color = extract_dominant_color(frame, person_bbox)
                    if not is_color_similar(registered_features, dominant_color):
                        draw_status_bar(frame, "ALERT: Unauthorized Access Detected!", color=(0, 0, 255))
                        alert_triggered = True

        # Save alert frames
        if alert_triggered:
            alert_frame_path = f"alert_{int(time.time())}.jpg"
            cv2.imwrite(alert_frame_path, frame)

        # Display detection counts
        cv2.putText(frame,
                    f"Scooters: {len(current_scooters)} | People: {len(current_people)}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    thickness=1)

        # Show video feed
        cv2.imshow("Scooter Theft Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Helper function to check proximity between two bounding boxes
def is_nearby(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2

    distance = np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
    return distance < 100

if __name__ == "__main__":
    main("../public/test_theft.mp4")  # Replace with your video file or camera feed
