from ultralytics import YOLO
 
model = YOLO('yolov13n.pt')   # Replace with the desired model scale
source = './ultralytics/assets/bus.jpg'  #更改为自己的图片路径
model.predict(source, save=True)  # 运行推理，并附加参数