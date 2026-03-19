from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="configs/kitti.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    name="Outdoor_Model"
)
