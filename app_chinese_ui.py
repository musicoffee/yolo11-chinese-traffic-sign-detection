import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import os
from pathlib import Path

# ================= 1. 配置中文映射字典 (48类) =================
CN_NAMES = {
    'i2': '正向行驶', 'i4': '立交直行', 'i5': '左转行驶', 'il100': '最低限速100', 'il60': '最低限速60',
    'il80': '最低限速80', 'io': '其它指示', 'ip': '允许停放', 'p10': '禁止货车', 'p11': '禁止鸣笛',
    'p12': '禁止拖车', 'p19': '禁止掉头', 'p23': '禁止车辆', 'p26': '禁止载客', 'p27': '禁止人力车',
    'p3': '禁止直行', 'p5': '禁止左转', 'p6': '禁止右转', 'pg': '禁止通行', 'ph4': '限高4m',
    'ph4.5': '限高4.5m', 'ph5': '限高5m', 'pl100': '限速100', 'pl120': '限速120', 'pl20': '限速20',
    'pl30': '限速30', 'pl40': '限速40', 'pl50': '限速50', 'pl60': '限速60', 'pl70': '限速70',
    'pl80': '限速80', 'pm20': '限重20t', 'pm30': '限重30t', 'pm55': '限重55t', 'pn': '禁止驶入',
    'pne': '禁止通行', 'po': '禁止停车', 'pr40': '限制宽度4m', 'ps': '停车让行', 'pw3': '限制轴重3t',
    'pw3.2': '限制轴重3.2t', 'pw4': '限制轴重4t', 'pw4.2': '限制轴重4.2t', 'w32': '陡坡',
    'w55': '施工', 'w57': '注意行人', 'w59': '注意儿童', 'wo': '其他警告'
}

# ================= 2. 加载模型 =================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "train" / "tt100k_final_exp" / "weights" / "best.pt"

if MODEL_PATH.exists():
    model = YOLO(str(MODEL_PATH))
    print(f"成功加载模型: {MODEL_PATH}")
else:
    print(f"错误：找不到模型文件 {MODEL_PATH}，请检查路径！")
    model = YOLO("yolo11m.pt")


# ================= 3. 定义中文绘制逻辑 =================
def draw_cn_box(image, box, label, color=(0, 255, 0)):
    """在图片上绘制中文标签"""
    x1, y1, x2, y2 = map(int, box)

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("msyh.ttc", 30)
    except:
        font = ImageFont.load_default()

    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

    text_bbox = draw.textbbox((x1, y1), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    bg_top = max(0, y1 - text_h - 10)
    bg_bottom = y1
    bg_right = x1 + text_w + 10

    draw.rectangle([x1, bg_top, bg_right, bg_bottom], fill=color)
    draw.text((x1 + 5, bg_top + 5), label, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ================= 4. 推理函数 =================
def predict_image(img, conf_threshold):
    if img is None:
        return None

    results = model.predict(source=img, conf=conf_threshold)
    res = results[0]
    canvas = res.orig_img.copy()

    for box in res.boxes:
        cls_id = int(box.cls[0])
        eng_name = model.names[cls_id]
        cn_name = CN_NAMES.get(eng_name, eng_name)
        conf = float(box.conf[0])
        label = f"{cn_name} {conf:.2f}"

        canvas = draw_cn_box(canvas, box.xyxy[0], label)

    return canvas


# ================= 5. Gradio 交互界面 =================
with gr.Blocks(title="路标识别系统 - YOLO11") as demo:
    gr.Markdown("# 🚗 中国路标实时识别系统 (YOLO11)")
    gr.Markdown("当前状态：已加载 TT100K 训练权重。支持 48 种常见路标识别。")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="上传图片")
            conf_slider = gr.Slider(0.1, 1.0, value=0.35, label="置信度阈值 (建议 0.3-0.5)")
            btn = gr.Button("开始识别", variant="primary")
        with gr.Column():
            output_img = gr.Image(type="numpy", label="识别结果")

    btn.click(predict_image, inputs=[input_img, conf_slider], outputs=output_img)

    gr.Markdown("### 使用说明")
    gr.Markdown("- 上传图片后点击【开始识别】即可看到结果。")
    gr.Markdown("- 如果画面太乱，请调高【置信度阈值】。")
    gr.Markdown("- 实时摄像头识别请在本地终端运行命令：`yolo predict model=runs/train/tt100k_final_exp/weights/best.pt source=0 show=True`")

if __name__ == "__main__":
    demo.launch(inbrowser=True)