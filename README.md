# YOLO11 Chinese Traffic Sign Detection

A Chinese traffic sign detection project based on **YOLO11** and the **TT100K** dataset.  
This repository includes dataset conversion, model training, result visualization, and a Gradio-based interactive inference interface.

## Features

- Convert the original **TT100K** dataset annotations into **YOLO format**
- Train a **YOLO11** model on selected Chinese traffic sign categories
- Visualize training loss and mAP performance
- Provide a **Gradio UI** for image-based traffic sign recognition
- Support **Chinese label display** for prediction results

---

## Project Structure

```bash
yolo11/
├── train.py                  # YOLO11 model training script
├── convert_tt100k.py         # Convert TT100K annotations to YOLO format
├── app_chinese_ui.py         # Gradio-based Chinese traffic sign recognition UI
├── visual.py                 # Training result visualization script
├── yolo11m.pt                # Initial pretrained YOLO11 weight
├── data/
│   └── tt100k_2021/          # Original TT100K dataset
├── datasets/
│   └── tt100k/
│       ├── images/           # Converted images
│       ├── labels/           # YOLO-format labels
│       └── data.yaml         # Dataset config file
└── runs/
    └── train/
        └── tt100k_final_exp/
            ├── weights/
            │   └── best.pt   # Best trained model
            ├── results.csv   # Training metrics
            └── my_report.png # Visualization result