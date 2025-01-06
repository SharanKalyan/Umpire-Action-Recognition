from ultralytics import YOLO

model = YOLO("./Weights/best.pt")
model.predict(source = "umpire.mp4",show=True,save=True , conf=0.5 , line_width = 1 , save_crop = True)