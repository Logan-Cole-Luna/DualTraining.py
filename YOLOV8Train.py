from ultralytics import YOLO
import wandb

# Log in to wandb
wandb.login(key="63033089206962899707825b6063aba50505b314")  # Replace with your wandb API key

# Initialize wandb
wandb.init(project="YoloV8__Combined_Aircraft", entity="logan03luna", name="Attempt1")


# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='CombinedDataset/data.yaml', epochs=100, imgsz=640)