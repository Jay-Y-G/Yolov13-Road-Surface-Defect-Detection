from ultralytics import YOLO


model = YOLO('./runs/detect/train/weights/best.pt')   # Replace with the desired model scale
source = './predict/4.jpg'  #更改为自己的图片路径
results=model.predict(source, save=True)  # 运行推理，并附加参数
