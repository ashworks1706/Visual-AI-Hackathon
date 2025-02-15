from roboflow import Roboflow
from ultralytics import YOLO

# Step 3: Download the first dataset (scooter-cycle-skatepark)
rf = Roboflow(api_key="j1wL1N5IgwOIiKW4RYuo")
# project1 = rf.workspace("medhaclass").project("scooter-cycle-skatepark")
# version1 = project1.version(2)
# dataset1 = version1.download("yolov8")

# # Step 4: Download the second dataset (escooter_detect)
# project2 = rf.workspace("cap2ej").project("escooter_detect")
# version2 = project2.version(1)
# dataset2 = version2.download("yolov8")

# Step 5: Load the YOLOv8n model
model = YOLO('yolov8n.pt')  # Ensure yolov8n.pt is in your working directory

# Step 6: Train on the first dataset
print("Training on scooter-cycle-skatepark dataset...")
model.train(data="scooter-cycle-skatepark/data.yaml", epochs=50, imgsz=640)

# Step 7: Train on the second dataset
print("Training on escooter_detect dataset...")
model.train(data="escooter_detect-1/data.yaml", epochs=50, imgsz=640)

print("Training completed for both datasets.")
