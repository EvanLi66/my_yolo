from ultralytics import YOLO

# Load a model
model = YOLO("/Users/carloslee/Desktop/yolo/ultralytics/runs/classify/train25/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx",batch=4)