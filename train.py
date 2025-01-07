from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11x-cls.yaml").load("/Users/carloslee/Desktop/yolo/pretrainweight/yolo11x-cls.pt")  # build a new model from YAML
# model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n-cls.yaml")  # build from YAML and transfer weights

# Train the model
# results = model.train(data="/Users/carloslee/Desktop/yolo/datasets/bevtrain/bevtest1920x1536", 
#                       epochs=2, 
#                       device='mps')
results = model.train(data="/Users/carloslee/Desktop/yolo/datasets/bevtrain/test_0107", 
                      epochs=2, 
                      batch =2,
                      device='mps')
results