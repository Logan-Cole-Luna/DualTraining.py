from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('YoloV8UnFroze/Attempt12/weights/best.pt')  # pretrained YOLOv8n model

# Define source as YouTube video URL
source = 'https://www.youtube.com/watch?v=UojjrMrW96g'

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    #result.save(filename='result.jpg')  # save to disk