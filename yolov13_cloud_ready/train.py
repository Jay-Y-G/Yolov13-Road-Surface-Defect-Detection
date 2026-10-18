import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
  model = YOLO('ultralytics/cfg/models/v13/yolov13n.yaml')
  model.load('yolov13n.pt')  #注释则不加载
  results = model.train(
    data='data.yaml',  #数据集配置文件的路径
    epochs=200,  #训练轮次总数，默认200
    batch=8,  #批量大小，即单次输入多少图片训练，默认16
    imgsz=640,  #训练图像尺寸
    workers=2,  #加载数据的工作线程数，默认8
    device= 0,  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
    optimizer='SGD',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
    amp= False,  #True 或者 False, 解释为：自动混合精度(AMP) 训练
    cache=False,  # True 在内存中缓存数据集图像，服务器推荐开启
    patience=50,  #早停轮数，默认50，训练过程中如果验证损失在patience轮内没有改善，则停止训练
)
  
  