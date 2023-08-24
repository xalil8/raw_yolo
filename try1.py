import torch
from models.experimental import attempt_load
import cv2
import numpy as np
import time

# Define your custom NMS function
def custom_nms(detections, conf_threshold, iou_threshold):
    # Implement your NMS algorithm here
    # 'detections' should be a list of bounding box predictions

    # Return the filtered bounding boxes after NMS
    pass

# Path to your custom .yaml configuration file
yaml_path = 'path_to_your_custom_model.yaml'

# Path to your pre-trained .pt checkpoint file

# Load your custom YOLOv5 model architecture
model = attempt_load("yolov5s.yaml", map_location="mps")

# Load the checkpoint
checkpoint = torch.load("v4.pt", map_location="mps")
model.load_state_dict(checkpoint['model'].float().state_dict())

# Set the model to evaluation mode
model.eval()

# Load your input image
"""input_image = 'path_to_input_image.jpg'
img = torch.from_numpy(cv2.imread(input_image)).float() / 255.0
img = img.permute(2, 0, 1).unsqueeze(0)  # Convert to CHW format and add batch dimension

# Perform inference
with torch.no_grad():
    detections = model(img)

# Apply your custom NMS algorithm
conf_threshold = 0.8
iou_threshold = 0.5
filtered_detections = custom_nms(detections, conf_threshold, iou_threshold)

# 'filtered_detections' contains the final bounding box predictions after NMS
# Process and visualize the detections as needed



video_cap = cv2.VideoCapture("v1.mp4")


while video_cap.isOpened():
    ret, frame = video_cap.read()
    if not ret:
        break


    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()
"""