import cv2
import numpy as np
from ultralytics import YOLO
from twilio.rest import Client
import face_recognition
import os
import time

# Initialize models
scooter_model = YOLO('yolov8n.pt')  # Base model
tool_model = YOLO('yolov8s-custom.pt')  # Custom trained on tools dataset
lock_model = YOLO('yolov8s-lock-segmentation.pt')  # Lock segmentation model

# Twilio setup
TWILIO_SID = 'your_account_sid'
TWILIO_TOKEN = 'your_auth_token'
client = Client(TWILIO_SID, TWILIO_TOKEN)

# Face recognition setup
known_face_encodings = []
known_face_names = []

def load_owner_profile(owner_image_path):
    global known_face_encodings, known_face_names
    owner_image = face_recognition.load_image_file(owner_image_path)
    owner_encoding = face_recognition.face_encodings(owner_image)[0]
    known_face_encodings.append(owner_encoding)
    known_face_names.append("Owner")

def send_alert(frame):
    filename = f"alert_{time.time()}.jpg"
    cv2.imwrite(filename, frame)
    message = client.messages.create(
        body='Scooter theft attempt detected!',
        from_='+1234567890',
        to='+0987654321',
        media_url=[filename]
    )
    os.remove(filename)

def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    scooter_parked = False
    owner_present = False
    lock_status = "locked"
    parked_position = None
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Scooter detection
        scooter_results = scooter_model(frame, classes=[67])  # Class 67 for scooter
        if scooter_results[0].boxes:
            scooter_box = scooter_results[0].boxes.xyxy[0].cpu().numpy()
            parked_position = scooter_box
            cv2.rectangle(frame, (int(scooter_box[0]), int(scooter_box[1])),
                         (int(scooter_box[2]), int(scooter_box[3])), (0,255,0), 2)

        # Face recognition for owner
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                owner_present = True
                scooter_parked = True

        # Tool detection near scooter
        if scooter_parked and not owner_present:
            tool_results = tool_model(frame, classes=[22, 44, 76])  # Classes for tools
            if tool_results[0].boxes:
                for box in tool_results[0].boxes.xyxy.cpu().numpy():
                    if parked_position and box_overlap(parked_position, box):
                        cv2.putText(frame, "THEFT ATTEMPT!", (10, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                        send_alert(frame)

        # Lock status detection
        lock_results = lock_model(frame)
        if lock_results[0].masks:
            lock_status = "intact" if check_lock_integrity(lock_results) else "broken"

        cv2.imshow('Scooter Security', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def box_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x1 < x2 and y1 < y2

def check_lock_integrity(lock_results):
    # Implement lock integrity check logic
    return True

if __name__ == "__main__":
    load_owner_profile("owner.jpg")
    main("parking_lot.mp4")
