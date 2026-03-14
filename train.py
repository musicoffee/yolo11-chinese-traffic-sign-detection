from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    # 项目根目录
    BASE_DIR = Path(__file__).resolve().parent

    # 数据配置文件路径
    DATA_YAML = BASE_DIR / "datasets" / "tt100k" / "data.yaml"

    # 1. 加载模型
    model = YOLO("yolo11m.pt")  # 使用预训练权重初始化

    # 2. 开始训练
    model.train(
        data=str(DATA_YAML),
        epochs=66,
        imgsz=1024,      # 提高分辨率以捕捉小型中国路标
        batch=4,
        workers=0,       # Windows 下避免共享内存问题
        device=0,        # GPU 加速
        project=str(BASE_DIR / "runs" / "train"),
        name='tt100k_final_exp',
        amp=True         # 开启混合精度加速
    )