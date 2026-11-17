"""Generate Mission 5 figures with the repository's original plotting tools.
"""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MISSION = Path(__file__).resolve().parent


def run_module(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT, check=True)


def main():
    experiments = sorted((ROOT / "training_results" / "joint_relational").glob("mission5_joint_*"))
    rows = []
    for experiment in experiments:
        summary_path = experiment / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["num_runs"] != 5:
            raise ValueError(f"Expected five completed seeds: {experiment}")
        seed_dir = experiment / "seed_42"
        output = MISSION / "figures" / experiment.name
        embedding_files = ("tsne.png", "pca_violin.png", "embedding_metrics.json",
                           "class_projection.csv", "embedding_samples.csv", "embeddings.npz")
        print(f"\nProcessing {experiment.name}", flush=True)
        if not all((output / name).is_file() for name in embedding_files):
            run_module("src.visualization.embedding_analysis",
                       "--model", "joint_relational",
                       "--checkpoint", seed_dir / "best_joint_relational_model.pth",
                       "--output-dir", output, "--seed", 42,
                       "--samples-per-class", 200, "--device", "cuda")
        confusion_dir = output / "confusion_matrix"
        if not all((confusion_dir / name).is_file() for name in
                   ("confusion_matrix_counts.png", "confusion_matrix_normalized.png",
                    "confusion_matrix_metrics.json")):
            run_module("src.visualization.confusion_matrix_analysis",
                       "--input", seed_dir / "digit_confusion_matrix.csv",
                       "--output-dir", confusion_dir,
                       "--title", experiment.name.replace("mission5_joint_", "Joint ") + " / seed 42")
        embedding = json.loads((output / "embedding_metrics.json").read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
        rows.append((config, metrics, embedding, output.relative_to(MISSION).as_posix()))

    lines = ["# 任务5：本模型实验结果", "",
             "准确率为五个随机种子的均值 ± 标准差，单位为百分比。",
             "embedding与混淆矩阵统一使用seed 42按关系验证准确率选出的最佳模型；",
             "embedding分析在测试集中每类抽取200张，共2000张，分析种子为42。",
             "ρ只描述第一主成分与数字类别的秩相关，不等同于分类准确率。", "",
             "| 关系对 | Shot | λ | 数字准确率(%) | 关系准确率(%) | seed42 ρ | Discovered order | 图目录 |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    for config, metrics, embedding, output in rows:
        digit = metrics["digit_test_accuracy"]
        relation = metrics["relation_test_accuracy"]
        order = " → ".join(map(str, embedding["discovered_order"]))
        lines.append(f"| {config['train_size']} | {config['shots_per_class']} | "
                     f"{config['classification_weight']} | {digit['mean']:.3f} ± {digit['std']:.3f} | "
                     f"{relation['mean']:.3f} ± {relation['std']:.3f} | "
                     f"{embedding['spearman_rho']:.4f} | {order} | [打开]({output}) |")
    lines.extend(["", "任务2、3、4的横向比较见 [统一对比报告](comparison/COMPARISON.md)。500k Siamese尚未提供。", ""])
    (MISSION / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCompleted {len(rows)} experiments. Summary: {MISSION / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
