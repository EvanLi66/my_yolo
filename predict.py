from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
# model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
model = YOLO("/Users/carloslee/Desktop/yolo/ultralytics/runs/classify/train45/weights/best.pt")  # build from YAML and transfer weights

# Train the model
results = model.predict(source="/Users/carloslee/Desktop/yolo/datasets/bevtrain/bevtest1920x1536/test/dirty", 
                        save=True,
                        device='mps')
# results = model.predict(source="/Users/carloslee/Desktop/yolo/datasets/bevtrain/bevtest1920\*1080/train/dirty/565.159099.jpg", 
#                         save=True,
#                         batch=4,
#                         device='mps')