from ultralytics import YOLO
 
#mytrain
model = YOLO('yolov8n.pt')
 
model.train(data="./cfg/datasets/hearb.yaml", epochs=200)
metrics = model.val()