import cv2
import numpy as np
import torch
from PIL import Image

# Load YOLOv5 model
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt',
                              # trust_repo=True,
                              # force_reload=True
                              )  # force_reload=True to update

# Set model to inference mode
# yolov5_model.eval()

# Open video capture device or file
cap = cv2.VideoCapture('test2.mp4')

# Loop over video frames
while cap.isOpened():
    # Read frame from video
    
    ret: bool
    frame: np.ndarray
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert OpenCV image to PIL image
    im: Image = Image.fromarray(frame)
    
    # Perform inference on image
    results = yolov5_model(im)
    
    # results.render()  # updates results.ims with boxes and labels
    
    # cv2.rectangle(frame, 100, 100, (100, 100), (0, 255, 0), 2)
    
    # Get bounding boxes and labels for detected objects
    bboxes = results.xyxy[0].cpu().numpy()
    labels = results.names[0]
    
    print(labels)
    print(len(bboxes))
    
    # Loop over detected objects and draw bounding boxes
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox[:4]
        print(x1, y1, x2, y2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display processed frame
    cv2.imshow('frame', frame)
    
    # Exit on ESC key press
    if cv2.waitKey(1) == 27:
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()
