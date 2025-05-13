from ultralytics import YOLO

# Load your PyTorch model
model = YOLO("my_mdoel.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'my_model.engine'
