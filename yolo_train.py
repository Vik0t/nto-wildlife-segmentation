from ultralytics import YOLO

model = YOLO("weights/yolo12n.pt")
results = model.train(data="data/data.yaml", epochs=25, imgsz=640)