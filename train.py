from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="coco128.yaml",
    epochs=5,
    imgsz=640,
    project="mlops",
    name="case-one",
    device="cuda",
)
