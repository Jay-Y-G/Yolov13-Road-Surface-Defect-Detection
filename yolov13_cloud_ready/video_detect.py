from ultralytics import YOLO

model = YOLO('./runs/train/weights/best.pt')

# 直接传入视频文件路径
results = model.predict(
    source='./predict/1.mp4',      # 视频文件路径
    conf=0.25,
    save=True,                   # 自动保存输出视频
    classes=[0,1,2],              # 可选：只显示你想要的类别
    show=True                    # 实时弹窗显示
)