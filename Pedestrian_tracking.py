import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
import cv2

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Load video or capture video from webcam
video_path = 'path_to_video_file.mp4'
cap = cv2.VideoCapture(video_path)

# Define class labels for detection
class_labels = ['person']

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        predictions = model(frame_tensor)
    
    # Process predictions
    for i, pred in enumerate(predictions[0]['labels']):
        if class_labels[pred] == 'person' and predictions[0]['scores'][i] > 0.5:
            bbox = predictions[0]['boxes'][i]
            bbox = bbox.to('cpu').numpy().astype(int)
            
            # Draw bounding box on frame
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Pedestrian Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
