import cv2
import time
from ultralytics import YOLO

# ===================== 配置区 =====================
# 修改为手机 IP Webcam 显示的地址（通常格式为 http://192.168.1.xxx:8080/video）
VIDEO_URL = "http://10.241.23.66:4444/video/mjpeg"

# 你训练好的模型路径
MODEL_PATH = r".\runs\train\weights/best.pt"

# 只显示这些类别（None = 显示所有；例如 [0,1,2] 只显示前3类）
SHOW_CLASSES = None
# ==================================================

# 加载模型
model = YOLO(MODEL_PATH)
print(f"模型加载完成，类别: {model.names}")

# 连接手机摄像头
cap = cv2.VideoCapture(VIDEO_URL)

if not cap.isOpened():
    print("❌ 无法连接手机摄像头！请检查：")
    print("   1. 手机 IP Webcam 是否已启动服务器")
    print("   2. 手机和电脑是否在同一 Wi-Fi 下")
    print(f"   3. 地址是否正确: {VIDEO_URL}")
    exit()

print("✅ 手机摄像头连接成功，按 q 键退出")

# 用于计算 FPS
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 读取视频帧失败，尝试重连...")
        time.sleep(0.5)
        continue

    # 模型推理
    results = model(frame, conf=0.25, classes=SHOW_CLASSES, verbose=False)

    # 在画面上绘制检测框
    annotated_frame = results[0].plot()

    # 计算并显示 FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示画面
    cv2.imshow("Phone Camera - Road Defect Detection", annotated_frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("程序已退出")