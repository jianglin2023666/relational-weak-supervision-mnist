"""Draw the implemented joint model as editable SVG and slide-ready PNG."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path(__file__).resolve().parent / "figures" / "model_architecture"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")
    dark, blue, orange = "#19334D", "#237EAF", "#C16B20"

    def text(x, y, s, size=14, color=dark, **kwargs):
        ax.text(x, y, s, ha="center", va="center", fontsize=size, color=color, **kwargs)

    def box(x, y, w, h, label, fill="#EDF5FA", edge=blue, size=14):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                                   linewidth=1.5, edgecolor=edge, facecolor=fill))
        text(x+w/2, y+h/2, label, size)

    def arrow(points, color=blue, dashed=False):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.6,
                    linestyle="--" if dashed else "-")
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=14, lw=1.6, color=color,
                                    linestyle="--" if dashed else "-"))

    text(8, 8.42, "联合监督模型与损失函数", 27, fontweight="bold")
    text(8, 7.91, "JointRelationalClassifier · 关系监督 + 极少量绝对类别标签", 15)

    # The repeated blocks denote calls to the very same modules, not separate models.
    ax.add_patch(FancyBboxPatch((3.18, 2.76), 4.43, 4.55,
                               boxstyle="round,pad=0.05,rounding_size=0.12",
                               linewidth=1.5, linestyle="--", edgecolor="#708DA4",
                               facecolor="#F8FBFD"))
    text(5.40, 7.03, "三次调用同一套参数 θ、φ", 14, fontweight="bold")
    for cy, label in [(6.23, "图片 A"), (4.79, "图片 B"), (3.35, "带标签图片 x")]:
        supervised = cy == 3.35
        color = orange if supervised else blue
        fill = "#FFF4E8" if supervised else "#EDF5FA"
        box(.45, cy-.40, 1.85, .8, label+"\n1 × 28 × 28", fill, color, 13)
        box(3.45, cy-.40, 1.73, .8, "共享骨干 fθ\n128维特征", fill, color, 13)
        box(5.57, cy-.40, 1.73, .8, "共享分类头 gφ\nLinear 128→10", fill, color, 12)
        arrow([(2.35,cy),(3.37,cy)],color)
        arrow([(5.22,cy),(5.50,cy)],color)
    box(7.98, 5.83, 1.70, .8, "Softmax\npA：10类概率", size=12)
    box(7.98, 4.39, 1.70, .8, "Softmax\npB：10类概率", size=12)
    arrow([(7.34,6.23),(7.90,6.23)])
    arrow([(7.34,4.79),(7.90,4.79)])
    box(10.18, 4.95, 2.18, 1.12,
        "固定大小关系公式\n"+r"$q=\sum_{i>j}p_A(i)p_B(j)$", size=13)
    arrow([(9.72,6.23),(9.92,6.23),(9.92,5.75),(10.10,5.75)])
    arrow([(9.72,4.79),(9.92,4.79),(9.92,5.24),(10.10,5.24)])
    box(12.92, 5.10, 2.55, .82, "关系损失 Lrelation\nBCE(q, r)", size=13)
    arrow([(12.40,5.51),(12.84,5.51)])
    text(14.20,6.66,"关系标签 r = 1[A > B]",12)
    arrow([(14.20,6.42),(14.20,6.00)])
    text(1.38,5.46,"关系对：50k / 500k",11)

    box(7.98, 2.95, 2.05, .8, "10类 logits\n不先做 Softmax", "#FFF4E8",orange,12)
    arrow([(7.34,3.35),(7.90,3.35)],orange)
    box(12.92, 2.94, 2.55, .82, "分类损失 Lclassification\nCrossEntropy(logits, y)",
        "#FFF4E8", orange, 11)
    arrow([(10.07,3.35),(12.84,3.35)],orange)
    text(11.41,3.78,"真实数字标签 y",12,color=orange)
    arrow([(11.41,3.62),(11.41,3.55),(12.84,3.55)],orange)
    text(1.38,2.63,"每类1或3张\n共10或30张，反复使用",11,color=orange)

    box(9.18, 1.45, 6.23, .91,
        r"$L_{total}=L_{relation}+\lambda L_{classification}$",
        "#ECF5EF", "#4E8462", 22)
    arrow([(14.90,5.06),(15.75,5.06),(15.75,2.07),(15.49,2.07)],dark)
    arrow([(14.20,2.90),(14.20,2.44)],orange)
    arrow([(9.10,1.90),(5.39,1.90),(5.39,2.66)],"#4E8462",True)
    text(5.14,1.50,"同一训练步反向传播，同时更新骨干 θ 与分类头 φ",12,color="#38654A")
    text(12.30,.99,"主实验 λ=0.5；权重消融 λ=0.1 / 0.5 / 1.0",12)
    text(8,.51,"骨干：Conv32 → Pool → Conv64 → Pool → Conv64 → Flatten → FC64 → Dropout(0.2) → FC128",11)
    text(8,.18,"按项目模型与训练代码重绘；测试数字分类时仅使用单张图片 → 共享骨干 → 分类头 → argmax。",10,color="#62778A")
    for ext in ("svg", "png"):
        path = OUT / f"joint_supervision_architecture.{ext}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
