"""Validate PR handoffs, compare models, and collect Mission 5 deliverables."""
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MISSION = Path(__file__).resolve().parent
ROOT = MISSION.parent
INPUT = MISSION / "comparison_inputs"
OUT = MISSION / "comparison"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix(path):
    rows = csv_read(path)
    return np.array([[int(row[f"pred_{i}"]) for i in range(10)] for row in rows])


def main():
    OUT.mkdir(exist_ok=True)
    records, checks, seed_rows = [], [], []

    def add(model, size, shot, lam, config_path, runs_path, summary_path,
            embedding_path, cm_pattern, metric_pattern, source):
        config, summary = read(config_path), read(summary_path)
        runs = csv_read(runs_path)
        seeds = [int(r["seed"]) for r in runs]
        assert sorted(seeds) == [42, 43, 44, 45, 46], runs_path
        stats = {}
        for metric in ["digit_test_accuracy", "relation_test_accuracy"]:
            if metric not in runs[0]:
                stats[metric] = None
                continue
            values = np.array([float(r[metric]) for r in runs])
            stats[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
            for key in ["mean", "std"]:
                assert math.isclose(stats[metric][key], summary["metrics"][metric][key], abs_tol=1e-6), (runs_path, metric)
        for run in runs:
            seed = int(run["seed"])
            cm_path = Path(str(cm_pattern).format(seed=seed))
            cm = matrix(cm_path)
            acc = cm.trace() / cm.sum() * 100
            assert cm.sum() == 10000 and math.isclose(acc, float(run["digit_test_accuracy"]), abs_tol=1e-7), cm_path
            seed_rows.append({"model": model, "relation_pairs": size, "shots": shot, "lambda": lam,
                              "seed": seed, "digit_accuracy_percent": float(run["digit_test_accuracy"]),
                              "relation_accuracy_percent": run.get("relation_test_accuracy", ""),
                              "best_epoch": run["best_epoch"]})
        emb = read(embedding_path)
        assert emb["num_samples"] == 2000 and emb["analysis_seed"] == 42
        record = {"model": model, "relation_pairs": size, "shots": shot, "lambda": lam,
                  "training_epochs": config.get("epochs", config.get("relation_epochs")),
                  "classifier_epochs": config.get("few_shot_epochs", ""),
                  "batch_size": config.get("batch_size", config.get("relation_batch_size")),
                  "digit_mean_percent": stats["digit_test_accuracy"]["mean"],
                  "digit_std_percentage_points": stats["digit_test_accuracy"]["std"],
                  "relation_mean_percent": stats["relation_test_accuracy"]["mean"] if stats["relation_test_accuracy"] else "",
                  "relation_std_percentage_points": stats["relation_test_accuracy"]["std"] if stats["relation_test_accuracy"] else "",
                  "seed42_spearman_rho": emb["spearman_rho"],
                  "seed42_discovered_order": ";".join(map(str, emb["discovered_order"])),
                  "pc1_variance_ratio": emb["pca_explained_variance_ratio"],
                  "source": source,
                  "config_path": str(Path(config_path).relative_to(MISSION)),
                  "embedding_path": str(Path(embedding_path).relative_to(MISSION))}
        records.append(record)
        checks.append({"model": model, "size": size, "shots": shot, "lambda": lam,
                       "five_seeds": True, "summary_recomputed": True, "five_confusion_matrices_match": True})

    t2 = INPUT / "task2/mission2_fh"
    add("Supervised", "none", "all", "", t2/"training_results/config.json", t2/"training_results/runs.csv",
        t2/"training_results/summary.json", t2/"figures/embedding_seed42/embedding_metrics.json",
        t2/"training_results/seed_{seed}/digit_confusion_matrix.csv", None, "PR #3 (100 epochs)")
    t3 = INPUT / "task3/mission3_why"
    for size in ["50k", "500k"]:
        add("Weakly supervised", size, 0, "", t3/f"summary/config_{size}.json", t3/f"summary/runs_{size}.csv",
            t3/f"summary/summary_{size}.json", t3/f"seed_42/embedding_{size}/embedding_metrics.json",
            t3/f"seed_{{seed}}/confusion_matrix_{size}.csv", None, "PR #2")
    t4 = INPUT / "task4/mission4_zdy"
    for shot in [1, 3]:
        base = t4 / f"{shot}shot"
        add("Siamese", "50k", shot, "", base/"training_results/config.json", base/"training_results/runs.csv",
            base/"training_results/summary.json", base/"embedding_seed42/embedding_metrics.json",
            base/"training_results/seed_{seed}/digit_confusion_matrix.csv", None, "PR #5")
    joint_dirs = sorted((ROOT/"training_results/joint_relational").glob("mission5_joint_*"))
    for exp in joint_dirs:
        if not (exp/"summary.json").is_file():
            continue
        dest = MISSION/"experiments"/exp.name
        dest.mkdir(parents=True, exist_ok=True)
        for p in exp.rglob("*"):
            if p.is_file() and p.suffix in [".csv", ".json"]:
                target = dest/p.relative_to(exp)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
        cfg = read(dest/"config.json")
        add("Joint", cfg["train_size"], cfg["shots_per_class"], cfg["classification_weight"],
            dest/"config.json", dest/"runs.csv", dest/"summary.json",
            MISSION/"figures"/exp.name/"embedding_metrics.json",
            dest/"seed_{seed}/digit_confusion_matrix.csv", None, "Mission 5 local")

    # Verify that few-shot comparisons use identical selected labelled samples.
    selected_checks = []
    for shot in [1, 3]:
        exp = next(p for p in joint_dirs if f"50k_{shot}shot_lambda0.5_" in p.name)
        for seed in [42,43,44,45,46]:
            a = read(exp/f"seed_{seed}/metrics.json")["selected_indices"]
            b = read(t4/f"{shot}shot/training_results/seed_{seed}/metrics.json")["selected_indices"]
            selected_checks.append({"shot":shot,"seed":seed,"identical_indices":a==b})

    sample_sets = []
    for record in records:
        samples = csv_read((MISSION/record["embedding_path"]).parent/"embedding_samples.csv")
        sample_sets.append([(row["dataset_index"], row["digit"]) for row in samples])
    same_embedding_samples = all(samples == sample_sets[0] for samples in sample_sets)
    (OUT/"validation.json").write_text(json.dumps({"experiments": checks, "few_shot_selection": selected_checks,
        "embedding_sample_indices_identical": same_embedding_samples},indent=2),encoding="utf-8")
    (OUT/"comparison.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    csv_write(OUT/"comparison.csv", records)
    csv_write(OUT/"per_seed.csv", seed_rows)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "svg.fonttype":"none", "font.size":11})

    def save(fig, name):
        for ext in ["png", "svg"]:
            fig.savefig(OUT/f"{name}.{ext}",dpi=190,bbox_inches="tight",facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(13,5.7),sharey=True)
    colors = {"Supervised":"#444444","Weakly supervised":"#B96C3F","Siamese":"#527EA1","Joint":"#3A805C"}
    baseline=records[0]
    for ax,size in zip(axes,["50k","500k"]):
        subset=[r for r in records if r["relation_pairs"]==size and (r["model"]!="Joint" or r["lambda"]==.5)]
        for x,r in enumerate(subset):
            vals=[s["digit_accuracy_percent"] for s in seed_rows if s["model"]==r["model"] and s["relation_pairs"]==size and s["shots"]==r["shots"] and s["lambda"]==r["lambda"]]
            ax.scatter(x+np.linspace(-.09,.09,5),vals,s=24,color=colors[r["model"]],alpha=.5)
            ax.errorbar(x,r["digit_mean_percent"],yerr=r["digit_std_percentage_points"],fmt="o",color=colors[r["model"]],capsize=5,lw=2)
        ax.axhline(baseline["digit_mean_percent"],color="#444444",ls="--",lw=1.2)
        ax.set_xticks(range(len(subset)),["基础弱监督" if r["model"]=="Weakly supervised" else f"{r['model']}\n{r['shots']}-shot" for r in subset])
        ax.set(title=f"{size}关系对",ylim=(0,104),ylabel="数字测试准确率 (%)")
        ax.grid(axis="y",alpha=.18)
    axes[1].text(.04,.13,"500k Siamese：尚未提供",transform=axes[1].transAxes,color="#666666")
    fig.suptitle("基础弱监督、Siamese与联合监督的跨种子结果",fontsize=17)
    fig.text(.5,.015,"点：种子42–46；误差线：样本标准差。虚线：100轮全监督参考。各方案训练预算不同。",ha="center",fontsize=11)
    fig.tight_layout(rect=(0,.07,1,.94))
    save(fig,"model_accuracy_comparison")

    # One representative seed for ordinal analysis, shared across all models.
    figs = [("全监督",t2/"figures/embedding_seed42"),
            ("基础弱监督 50k",t3/"seed_42/embedding_50k"),
            ("Siamese 50k（预训练骨干）",t4/"1shot/embedding_seed42"),
            ("联合监督 50k / 1-shot",MISSION/"figures"/next(p.name for p in joint_dirs if "50k_1shot_lambda0.5_" in p.name))]
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for col,(title,base) in enumerate(figs):
        emb=read(base/"embedding_metrics.json")
        for row,name in enumerate(["tsne.png","pca_violin.png"]):
            axes[row,col].imshow(plt.imread(base/name))
            axes[row,col].axis("off")
        axes[0,col].set_title(title,fontsize=13)
        axes[1,col].set_title(f"seed42 ρ = {emb['spearman_rho']:.3f}",fontsize=12)
    fig.suptitle("四种方案的特征分布与PC1顺序相关性",fontsize=19)
    fig.text(.5,.02,"每类200张测试图，分析种子42。ρ不等同于分类准确率。Siamese的图来自冻结分类前的关系预训练骨干。",ha="center",fontsize=11)
    fig.tight_layout(rect=(0,.05,1,.95))
    save(fig,"embedding_comparison")

    # A clearly labelled failure-case diagnostic complements the fixed seed42 gallery.
    joint50=next(p for p in joint_dirs if "50k_1shot_lambda0.5_" in p.name)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,path,title in [(axes[0],t3/"seed_46/confusion_matrix_50k.csv","基础弱监督 / seed46"),
                          (axes[1],joint50/"seed_46/digit_confusion_matrix.csv","联合监督1-shot / seed46")]:
        cm=matrix(path)
        normalized=cm/cm.sum(axis=1,keepdims=True)
        ax.imshow(normalized,cmap="Blues",vmin=0,vmax=1)
        for i in range(10):
            for j in range(10):
                if normalized[i,j]>=.05:
                    ax.text(j,i,f"{normalized[i,j]*100:.0f}",ha="center",va="center",fontsize=9,
                            color="white" if normalized[i,j]>.5 else "black")
        ax.set(xticks=range(10),yticks=range(10),xlabel="预测数字",ylabel="真实数字",
               title=f"{title}\n数字准确率 {cm.trace()/cm.sum()*100:.2f}%")
    fig.suptitle("类别错位诊断：50k条件下的同一随机种子",fontsize=16)
    fig.text(.5,.01,"选seed46用于展示基础弱监督的失败案例，不能代替五种子的总体统计。格内数值为行内百分比。",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.94))
    save(fig,"misalignment_seed46")

    lines=["# 四种监督方案统一对比", "", "数据取自任务2 PR #3、任务3 PR #2、任务4 PR #5及本地任务5实验。全部结果为种子42～46的均值 ± 样本标准差（ddof=1）。准确率单位为%，标准差单位为百分点。PR版本和文件哈希见comparison_inputs/download_manifest.json。", "",
           "| 模型 | 关系对 | 绝对标签 | 训练预算 | 数字准确率 | 关系准确率 | seed42 ρ |",
           "|---|---|---|---|---|---|---|"]
    for r in records:
        label="全量50,000张" if r["shots"]=="all" else f"每类{r['shots']}张"
        budget=f"{r['training_epochs']}轮"+(f" + 分类头{r['classifier_epochs']}轮" if r["classifier_epochs"] else "")
        relation="不适用" if r["relation_mean_percent"]=="" else f"{r['relation_mean_percent']:.3f} ± {r['relation_std_percentage_points']:.3f}"
        name=r["model"]+(f" λ={r['lambda']}" if r["model"]=="Joint" else "")
        lines.append(f"| {name} | {r['relation_pairs']} | {label} | {budget} | {r['digit_mean_percent']:.3f} ± {r['digit_std_percentage_points']:.3f} | {relation} | {r['seed42_spearman_rho']:.4f} |")
    lines.extend(["", "## 实验设置与可比性", "",
                  "- 任务2当前有效版本为100轮全监督，batch size=64。旧5轮PR已关闭，不再混用。",
                  "- 基础弱监督和联合监督均为5轮，关系batch size=128。Siamese为5轮关系预训练，再冻结骨干训练分类头100轮。",
                  "- 联合监督每个关系批次都重复使用少标签集合，50k下约1955次分类更新，500k下约19535次。Siamese少标签集合每轮仅一个批次，共100次分类更新。不是等更新次数的控制实验。",
                  f"- 对比Siamese与联合监督，1/3-shot五个种子的已标注样本索引全部一致：{all(x['identical_indices'] for x in selected_checks)}。该核对不意味着硬件与数值实现完全相同。",
                  f"- 各组seed42的embedding分析测试样本索引与类别顺序一致：{same_embedding_samples}，每组2000张。",
                  "- 同seed并不保证不同硬件、PyTorch版本下训练轨迹相同。任务2/4在CPU，任务3/5在GPU。",
                  "- 全监督使用完整数字标签，关系方案额外使用大小关系标签。少量绝对标签不等于完全无标签，关系对生成也依赖MNIST类别信息。",
                  "- 500k Siamese尚未提供，明确记为缺失，不外推或补造数据。",
                  "- checkpoint选择不同：全监督按数字验证准确率，关系模型按关系验证准确率，Siamese分类头使用固定末轮。",
                  "", "## 当前数据支持的结论", "",
                  "1. 50k、1-shot联合监督为98.162±0.186%，高于当前Siamese的96.132±0.784%，差2.030个百分点。3-shot分别为98.094±0.141%与96.920±0.131%，差1.174个百分点。该结论限定于当前实验方案，训练预算不同。",
                  "2. 基础弱监督50k/500k数字准确率分别为64.034±31.912%、53.166±26.121%，而联合监督主实验稳定在98%～99%左右。结合seed46混淆矩阵，少量绝对标签可缓解本次实验中的类别错位。",
                  "3. 500k、1-shot联合监督为99.136±0.077%，与100轮全监督99.292±0.077%相差0.156个百分点。可以表述为当前设置下接近全监督参考，不能称为相同训练预算下等效。",
                  "4. λ=0.1/0.5/1.0数字准确率均在98%左右。λ控制少标签分类损失的贡献，权重越大并不保证测试效果越好。没有显著性检验，保留默认λ=0.5作为主结果。",
                  "5. 联合监督的PC1 ρ在本次seed42结果中低于基础弱监督和Siamese，因此不能声称它改善了线性顺序相关性。分类更准与PC1更有序是不同目标。",
                  "", "## 图及使用方式", "",
                  "- model_accuracy_comparison.png/svg：五seed数字准确率与标准差，虚线为100轮全监督参考。",
                  "- embedding_comparison.png/svg：固定seed42的t-SNE与PCA汇总。原图和完整统计均保留。",
                  "- misalignment_seed46.png/svg：明确标注的错位诊断案例，配合总体统计解释。",
                  "", "## 各模型的seed42 Discovered order", ""])
    for r in records:
        lines.append(f"- {r['model']} / {r['relation_pairs']} / shot={r['shots']} / λ={r['lambda']}："+r["seed42_discovered_order"].replace(";"," → "))
    (OUT/"COMPARISON.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # Keep a reproducible snapshot of the actual task5 implementation.
    for rel in ["src/models/base_model.py", "src/models/joint_relational_classifier.py",
                "src/training/train_joint_relational.py", "src/utils/experiment_utils.py",
                "src/visualization/embedding_analysis.py", "src/visualization/confusion_matrix_analysis.py",
                "data/dataset.py", "data/mnist/download_MNIST.py"]:
        dest=MISSION/"source_snapshot"/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,dest)
    print(f"Validated {len(records)} experiments, {len(seed_rows)} seed results and {len(seed_rows)} confusion matrices.")
    print(f"Labelled indices identical across task4/5: {all(x['identical_indices'] for x in selected_checks)}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
