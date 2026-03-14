import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path

# --- 配置信息 ---
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR / "data" / "tt100k_2021"      # 原始数据路径
SAVE_DIR = BASE_DIR / "datasets" / "tt100k"       # 转换后存放路径
JSON_FILE = ROOT_DIR / "annotations_all.json"

# 为了作业效果，我们筛选样本数大于100的类别（TT100K的经典做法）
SELECTED_CLASSES = [
    'i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'io', 'ip', 'p10', 'p11', 'p12', 'p19', 'p23', 'p26',
    'p27', 'p3', 'p5', 'p6', 'pg', 'ph4', 'ph4.5', 'ph5', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40',
    'pl50', 'pl60', 'pl70', 'pl80', 'pm20', 'pm30', 'pm55', 'pn', 'pne', 'po', 'pr40', 'ps', 'pw3',
    'pw3.2', 'pw4', 'pw4.2', 'w32', 'w55', 'w57', 'w59', 'wo'
]


def convert():
    # 1. 创建文件夹结构
    for split in ['train', 'val', 'test']:
        os.makedirs(SAVE_DIR / 'images' / split, exist_ok=True)
        os.makedirs(SAVE_DIR / 'labels' / split, exist_ok=True)

    # 2. 读取JSON
    print("正在加载标注文件...")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    imgs = data['imgs']

    # 类别映射表
    cls_map = {name: i for i, name in enumerate(SELECTED_CLASSES)}

    # 3. 开始转换
    print("正在转换格式并复制文件...")
    for img_id, info in tqdm(imgs.items()):
        path = info['path']

        # TT100K 原始通常只有 train/test，这里把非 train 的先放到 val
        if 'train' in path:
            split = 'train'
        else:
            split = 'val'

        # 筛选出我们需要的目标类别
        valid_objs = []
        for obj in info['objects']:
            if obj['category'] in SELECTED_CLASSES:
                valid_objs.append(obj)

        if not valid_objs and split == 'train':
            continue

        # 复制图片
        src_img_path = ROOT_DIR / path
        dst_img_path = SAVE_DIR / 'images' / split / f"{img_id}.jpg"
        if not dst_img_path.exists():
            shutil.copy(src_img_path, dst_img_path)

        # 生成 YOLO 标签
        img_w, img_h = 2048, 2048  # TT100K 固定尺寸
        label_path = SAVE_DIR / 'labels' / split / f"{img_id}.txt"

        with open(label_path, 'w', encoding='utf-8') as f:
            for obj in valid_objs:
                cls_id = cls_map[obj['category']]
                bbox = obj['bbox']

                x_center = (bbox['xmin'] + bbox['xmax']) / 2.0 / img_w
                y_center = (bbox['ymin'] + bbox['ymax']) / 2.0 / img_h
                w = (bbox['xmax'] - bbox['xmin']) / img_w
                h = (bbox['ymax'] - bbox['ymin']) / img_h

                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # 4. 生成 data.yaml
    yaml_content = f"path: {SAVE_DIR.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n"
    for i, name in enumerate(SELECTED_CLASSES):
        yaml_content += f"  {i}: {name}\n"

    with open(SAVE_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"恭喜！数据处理完成，存放于: {SAVE_DIR}")


if __name__ == "__main__":
    convert()