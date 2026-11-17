"""Redraw Figure 3(a)'s layout and add the implemented joint supervision path."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures" / "model_architecture"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = torch.load(ROOT / "data/mnist/supervised/x_train.pt", weights_only=True)
    labels = torch.load(ROOT / "data/mnist/supervised/y_train.pt", weights_only=True)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 16), ylim=(0, 10))
    ax.axis("off")
    dark, blue, green, orange = "#233544", "#4B95D6", "#3C7F34", "#C67628"

    def txt(x, y, s, fs=14, color=dark, **kwargs):
        ax.text(x,y,s,fontsize=fs,ha="center",va="center",color=color,**kwargs)

    def box(cx,cy,w,h,s,edge="#8C939A",fill="white",fs=14):
        ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
            boxstyle="round,pad=0.03,rounding_size=0.07",facecolor=fill,edgecolor=edge,lw=1.8))
        txt(cx,cy,s,fs)

    def arrow(points,color="#246982",dashed=False):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.7,ls="--" if dashed else "-")
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",mutation_scale=15,
                                    lw=1.7,color=color,linestyle="--" if dashed else "-"))

    txt(8,9.51,"任务5：关系监督 + 少量类别标签的联合训练",25,fontweight="bold")
    txt(8,9.04,"参照论文 Figure 3(a) 重绘；右侧橙色分支为本项目任务5的扩展",14)

    # Retain the paper's paired vertical branches and central relational function.
    ax.add_patch(FancyBboxPatch((.65,2.88),8.1,4.43,boxstyle="round,pad=0.06,rounding_size=0.16",
                                facecolor="#F9FCF8",edgecolor=green,lw=2))
    ax.add_patch(FancyBboxPatch((10.10,2.88),5.22,4.43,boxstyle="round,pad=0.06,rounding_size=0.16",
                                facecolor="#FFFBF5",edgecolor=orange,lw=2))
    txt(4.70,7.18,"关系监督分支：沿用论文结构",14,color=green,fontweight="bold")
    txt(12.71,7.18,"任务5新增：少标签分类分支",14,color=orange,fontweight="bold")

    for cx, digit, name in [(2.65,7,"图片 A"),(6.75,3,"图片 B"),(12.70,8,"带标签图片 x")]:
        idx = int((labels == digit).nonzero()[0].item())
        ax.imshow(images[idx].numpy(),cmap="gray",vmin=0,vmax=255,
                  extent=(cx-.36,cx+.36,7.92,8.64),zorder=3,interpolation="nearest")
        txt(cx,7.64,name+"：1 × 28 × 28",12)
        arrow([(cx,7.43),(cx,6.81)])
        box(cx,6.45,3.08,.65,"BaseModel\n128维 embedding",blue,fs=13)
        arrow([(cx,6.09),(cx,5.79)])
        box(cx,5.45,3.08,.60,"Fully-connected linear (10)",green,fs=11)
        if cx < 10:
            arrow([(cx,5.12),(cx,4.53)])
            txt(cx+.78,4.83,"Softmax",12)
            box(cx,4.18,3.08,.60,"Class probabilities：10",fs=12)
        else:
            arrow([(cx,5.12),(cx,4.53)],orange)
            box(cx,4.18,3.08,.60,"10类 logits（直接送入交叉熵）",orange,fs=11)

    ax.plot([4.23,5.17],[6.45,6.45],ls="--",color=blue,lw=1.4)
    ax.plot([8.32,11.12],[6.45,6.45],ls="--",color=blue,lw=1.4)
    ax.plot([4.23,5.17],[5.45,5.45],ls="--",color=green,lw=1.4)
    ax.plot([8.32,11.12],[5.45,5.45],ls="--",color=green,lw=1.4)
    # White-backed shared labels keep dashed connections legible.
    for x,y,s,c in [(9.42,6.45,"共享\nθ",blue),(9.42,5.45,"共享\nφ",green)]:
        ax.text(x,y,s,ha="center",va="center",fontsize=12,color=c,
                bbox=dict(facecolor="white",edgecolor="none",pad=3))

    box(4.70,3.31,5.70,.60,r"$q=P(A>B)=\sum_{i>j}p_A(i)p_B(j)$",green,fs=15)
    arrow([(2.65,3.85),(2.65,3.68)])
    arrow([(6.75,3.85),(6.75,3.68)])
    box(4.70,2.18,5.70,.65,"关系损失：Lrelation = BCE(q, r)",green,fs=14)
    arrow([(4.70,2.97),(4.70,2.54)])
    txt(9.08,2.22,"r = 1[A > B]\n示例：7 > 3，r=1",10,color=green)
    arrow([(7.98,2.18),(7.63,2.18)])

    box(12.70,3.31,4.56,.60,"分类损失：CrossEntropy(logits, y)",orange,fs=12)
    arrow([(12.70,3.85),(12.70,3.68)],orange)
    txt(12.70,2.42,"示例标签 y=8\n每类1/3张，共10/30张，训练中反复使用",12,color=orange)

    box(8.00,1.13,8.80,.81,
        r"$L_{total}=L_{relation}+\lambda\,L_{classification}$","#587264","#F0F7F2",23)
    arrow([(4.70,1.82),(4.70,1.59)],green)
    arrow([(15.02,3.31),(15.64,3.31),(15.64,1.13),(12.48,1.13)],orange)
    txt(8,.50,"同一步反向传播，同时更新共享骨干 θ 和共享分类头 φ；主实验 λ=0.5",13)
    txt(8,.15,"数字图片来自MNIST，仅作输入示意；结构与数据流依据本项目 JointRelationalClassifier 及训练代码。",10,color="#647480")
    for ext in ("svg","png"):
        p=OUT/f"joint_supervision_paper_style.{ext}"
        fig.savefig(p,dpi=200,facecolor="white")
        print(p)
    plt.close(fig)


if __name__ == "__main__":
    main()
