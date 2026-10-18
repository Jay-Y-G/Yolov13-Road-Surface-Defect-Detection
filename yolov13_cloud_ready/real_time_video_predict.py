import cv2
from ultralytics import YOLO

model = YOLO('./best.pt')

# 打开视频文件
cap = cv2.VideoCapture('你的视频.mp4')

# 获取视频信息，用于保存输出视频
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(
    'output_video.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()      # 逐帧读取
    if not ret:
        break                    # 视频读完退出

    # 对当前帧进行推理
    results = model(frame, conf=0.25, classes=[2, 3])

    # 在帧上绘制检测框
    annotated_frame = results[0].plot()

    # 保存到输出视频
    out.write(annotated_frame)

    # 实时显示
    cv2.imshow('Road Defect Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()