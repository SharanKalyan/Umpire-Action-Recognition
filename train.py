from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(data = "data.yaml", imgsz = 640,
	batch = 4 , epochs = 25, workers = 0 , device = 0)