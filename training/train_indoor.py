from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="configs/sunrgbd.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    name="Indoor_Model"
)
