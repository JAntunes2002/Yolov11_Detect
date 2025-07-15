from ultralytics import YOLO

model = YOLO("yolo11m")

model.train(data = "data.yaml", imgsz =640, batch = 16, epochs = 100, workers = 0, device = 0)
