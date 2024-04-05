from ultralytics import YOLO
import wandb

import torch
import torchvision



def setup_and_train():
    print("PyTorch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA is available:", torch.cuda.is_available())
    print("CUDA device name:", torch.cuda.get_device_name(0))

    # Check if a CUDA GPU is available and set PyTorch to use it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    # Load a model
    model = YOLO('yolov8n.pt')  # Specify the device here

    # Train the model
    results = model.train(data='CombinedDataset/data.yaml', epochs=100, imgsz=640)
    return results

if __name__ == '__main__':
    results = setup_and_train()
