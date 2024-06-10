from ultralytics import YOLO

# Load a model
model = YOLO('YoloV8UnFroze/AttemptCivilianStart/weights/best.pt')  # pretrained YOLOv8n model
#model = YOLO('YoloV8Civillian/Attempt2/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['img.png', 'CombinedDataset/train/images/1570250.jpg', 'img_1.png', 'img_2.png'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk