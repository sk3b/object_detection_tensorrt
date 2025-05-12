from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("my_model.pt")

# Export the model to TensorRT with DLA enabled (only works with FP16 or INT8)
model.export(format="engine")  # dla:0 or dla:1 corresponds to the DLA cores

# Load the exported TensorRT model
trt_model = YOLO("my_model.engine")
