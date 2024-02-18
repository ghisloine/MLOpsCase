from ultralytics import YOLO

# Load a model
model = YOLO("mymodels/best.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="coco128.yaml",
    epochs=10,
    imgsz=736,
    project="mlops",
    name="case-two",
    device="cuda",
)
