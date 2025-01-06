from ultralytics import YOLO

# Load a model
model = YOLO("/Users/carloslee/Desktop/yolo/ultralytics/runs/classify/train/weights/best.pt")

# Validate with a custom dataset
metrics = model.val(data="/Users/carloslee/Desktop/yolo/datasets/bevtrain/bevtest1920x1536",
                    batch =1,
                    )
metrics
# print(metrics.box.map)  # map50-95