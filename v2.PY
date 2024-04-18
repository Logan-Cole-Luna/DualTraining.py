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
    results = model.train(data='CombinedDataset/data.yaml', epochs=300, imgsz=640)
    return results

def export_to_onnx(model, input_size=(1, 3, 640, 640), path='model.onnx'):
    # Create a dummy input tensor matching the input size
    dummy_input = torch.randn(*input_size, device='cuda' if torch.cuda.is_available() else 'cpu')
    # Export the model
    torch.onnx.export(model, dummy_input, path, opset_version=11)
    print(f"Model exported to ONNX format at {path}")



if __name__ == '__main__':
    results = setup_and_train()
    export_to_onnx(results.model, path='final_model.onnx')
    # Assuming the model is in evaluation mode and ready for inference
    results.model.eval()
