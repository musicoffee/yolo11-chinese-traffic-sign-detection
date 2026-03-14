import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ================= 1. 配置路径 =================
BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "runs" / "train" / "tt100k_final_exp"
RESULTS_CSV = RESULT_DIR / "results.csv"
SAVE_PATH = RESULT_DIR / "my_report.png"


# ================= 2. 解决中文乱码问题 =================
def set_ch_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


def generate_visual_report():
    set_ch_font()

    if not RESULTS_CSV.exists():
        print(f"错误：找不到文件 {RESULTS_CSV}，请检查路径是否正确！")
        return

    save_dir = SAVE_PATH.parent
    if not save_dir.exists():
        os.makedirs(save_dir)
        print(f"提示：已自动创建目录 {save_dir}")

    df = pd.read_csv(RESULTS_CSV)
    df.columns = [c.strip() for c in df.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(df['epoch'], df['train/box_loss'], label='定位损失 (Box)', linewidth=2)
    ax1.plot(df['epoch'], df['train/cls_loss'], label='分类损失 (Cls)', linewidth=2)
    ax1.set_title("模型学习进度 (Loss 下降)", fontsize=14, pad=15)
    ax1.set_xlabel("训练轮次 (Epoch)")
    ax1.set_ylabel("损失值")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    map50_col = 'metrics/mAP50(B)' if 'metrics/mAP50(B)' in df.columns else 'metrics/mAP_0.5'
    map95_col = 'metrics/mAP50-95(B)' if 'metrics/mAP50-95(B)' in df.columns else 'metrics/mAP_0.5:0.95'

    ax2.plot(df['epoch'], df[map50_col], label='mAP50 (基础准确率)', linewidth=3)
    ax2.plot(df['epoch'], df[map95_col], label='mAP50-95 (严苛定位精度)', linewidth=2)

    final_map50 = df[map50_col].iloc[-1]
    ax2.annotate(
        f'最终成绩: {final_map50:.3f}',
        xy=(df['epoch'].iloc[-1], final_map50),
        xytext=(df['epoch'].iloc[-1] - 10, final_map50 - 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05)
    )

    ax2.set_title("模型识别表现 (mAP 上升)", fontsize=14, pad=15)
    ax2.set_xlabel("训练轮次 (Epoch)")
    ax2.set_ylabel("准确率 (0-1)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    try:
        plt.savefig(SAVE_PATH, dpi=300)
        print(f"可视化报告已成功生成：{SAVE_PATH}")
    except Exception as e:
        print(f"保存图片失败: {e}")

    plt.show()


if __name__ == "__main__":
    generate_visual_report()