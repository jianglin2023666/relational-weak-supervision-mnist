"""Run task 2 with the original configuration and package reproducible results."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'configs').is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('TQDM_MININTERVAL', '30')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=ROOT / 'task2_outputs')
    parser.add_argument('--experiment', type=Path, help='Reuse an already completed experiment for plotting only.')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4)
    from data.mnist.download_MNIST import download_mnist
    from src.training.train_supervised import run_experiment
    from src.visualization.embedding_analysis import run_analysis
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    download_mnist()
    if args.experiment:
        experiment = args.experiment.resolve()
        summary = json.loads((experiment / 'summary.json').read_text(encoding='utf-8'))
    else:
        experiment, summary = run_experiment(ROOT / 'configs/supervised.json')
        experiment = Path(experiment).resolve()
    (args.output_root / 'latest_experiment.txt').write_text(str(experiment), encoding='utf-8')
    bundle = args.output_root / ('mission2_results_' + experiment.name)
    bundle.mkdir(exist_ok=True)
    shutil.copytree(experiment, bundle / 'training_results', dirs_exist_ok=True)
    figures = bundle / 'figures'
    figures.mkdir(exist_ok=True)
    embedding = run_analysis(
        model_name='supervised',
        checkpoint_path=experiment / 'seed_42/best_supervised_model.pth',
        output_dir=figures / 'embedding_seed42',
        samples_per_class=200, batch_size=256, seed=42, num_workers=0,
        requested_device='auto', perplexity=30.0,
    )
    subprocess.run([sys.executable, '-m', 'src.visualization.confusion_matrix_analysis',
                    '--input', str(experiment / 'seed_42/digit_confusion_matrix.csv'),
                    '--output-dir', str(figures / 'confusion_matrix_seed42'),
                    '--title', 'Supervised - Seed 42'], check=True)
    runs = list(csv.DictReader((experiment / 'runs.csv').open(encoding='utf-8-sig')))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in runs:
        history = list(csv.DictReader((experiment / f"seed_{run['seed']}/history.csv").open(encoding='utf-8-sig')))
        axes[0].plot([int(r['epoch']) for r in history],
                     [float(r['val_digit_accuracy']) for r in history], label=f"Seed {run['seed']}")
    axes[0].set(xlabel='Epoch', ylabel='Validation accuracy (%)', title='Validation accuracy by seed')
    axes[0].legend()
    axes[1].bar([r['seed'] for r in runs], [float(r['digit_test_accuracy']) for r in runs])
    axes[1].set(xlabel='Seed', ylabel='Test accuracy (%)', title='Five-seed digit test accuracy', ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(figures / 'five_seed_results.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.set_axis_off()
    labels = ['MNIST image\n[B,1,28,28]', 'Shared BaseModel\n128-D embedding', 'Linear(128,10)\n10 logits', 'Cross-entropy\nTrue digit label']
    for i, label in enumerate(labels):
        ax.text(.12 + i*.25, .55, label, ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.7', facecolor='#dceaff', edgecolor='#3564a0'))
        if i < 3:
            ax.annotate('', xy=(.29+i*.25,.55), xytext=(.21+i*.25,.55), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(.5,.12,'Train: update backbone + classifier. Inference: argmax(logits) -> digit 0-9.',
            ha='center', transform=ax.transAxes)
    fig.savefig(figures / 'supervised_training_flow.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    config = json.loads((experiment / 'config.json').read_text())
    provenance = {'python': sys.version, 'torch': torch.__version__,
                  'threads': torch.get_num_threads(), 'experiment': str(experiment.relative_to(ROOT)),
                  'commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  'config': config, 'embedding_seed': 42, 'embedding_samples_per_class': 200}
    (bundle / 'provenance.json').write_text(json.dumps(provenance,indent=2), encoding='utf-8')
    subprocess.run([sys.executable,'-m','pip','freeze'], stdout=(bundle / 'requirements-lock.txt').open('w'), check=True)
    stats = summary['metrics']['digit_test_accuracy']
    rows = '\n'.join(f"| {r['seed']} | {r['best_epoch']} | {float(r['digit_test_accuracy']):.2f}% |" for r in runs)
    report = f'''# 任务2：全监督 MNIST 基线实验

数字测试准确率：**{stats['mean']:.2f} ± {stats['std']:.2f}%**（五个 seed，样本标准差 ddof=1）。

| Seed | 最佳 epoch（按验证集选择） | Digit test accuracy |
|---|---|---|
{rows}

## 实验设置
训练/验证/测试分别为 50,000 / 10,000 / 10,000 张图像；每个 seed 训练 {config['epochs']} 轮。
batch size={config['batch_size']}，Adam，学习率={config['learning_rate']}，weight decay={config['weight_decay']}。
使用原项目 BaseModel + Linear(128,10)，交叉熵同时更新骨干和分类头。
每个 seed 的模型均按验证准确率保存最佳权重，测试集只用于最终评价。
任务2使用全部数字标签；50k/500k关系对是关系监督任务的数据量设置，不能当成两套不同的全监督数据。

## 图表与 embedding 分析
- `figures/supervised_training_flow.png`：训练结构图。
- `figures/five_seed_results.png`：五 seed 验证曲线和测试准确率。
- `figures/confusion_matrix_seed42/`：seed 42 原始计数及归一化混淆矩阵。
- `figures/embedding_seed42/`：t-SNE、PCA小提琴图及原始 CSV/NPZ/JSON。
- Spearman ρ = {embedding['spearman_rho']:.6f}。
- Discovered order = {' → '.join(map(str,embedding['discovered_order']))}。
- embedding 固定 seed 42、每类 200 张测试图片，不按测试准确率挑选 seed。

全监督基线提供相同骨干下使用完整标签的性能参考，不代表其他方法绝不可能超过的数学上界。
交叉熵目标鼓励类别可分，但不要求 PCA 第一主成分按 0 到 9 排序，因此分类准确率和 PCA 序关系相关性衡量不同性质。
t-SNE仅用于观察局部类别分布，不能据此证明全局距离或数字顺序。

## 提供给任务5的数据
`training_results/runs.csv`、`summary.json`、`config.json`、`metadata.json`，以及 embedding 的 JSON/CSV 和混淆矩阵。
准确率单位为百分数。建议对比时保留训练数据量、seed、shot 和设备信息。

## PPT建议（两页）
1. 全监督结构图、训练设置、五seed准确率及 mean ± std、混淆矩阵。
2. t-SNE、PCA小提琴图、Spearman ρ和 discovered order；解释分类可分性与序关系的区别。

## 复现与提交
使用项目根目录的 `run_task2.cmd` 一键启动；结果包中的脚本为备份，复现时先复制回项目根目录。
在项目根目录执行 `.venv/Scripts/python.exe run_task2.py --experiment <已完成实验目录>` 可只重建图表和结果包。
本文件夹含五seed最佳权重和统计数据，不含MNIST原始数据或虚拟环境。
提交PR前把文件夹按团队约定改为 `mission2_你的姓名缩写`。本次未提交PR，也未编辑共享PPT。
'''
    (bundle / '实验报告.md').write_text(report, encoding='utf-8')
    shutil.copy2(__file__, bundle / 'run_task2.py')
    shutil.copy2(Path(__file__).with_suffix('.cmd'), bundle / 'run_task2.cmd')
    shutil.copy2(ROOT / 'requirements.txt', bundle / 'requirements.txt')
    shutil.copy2(ROOT / 'requirements-visualization.txt', bundle / 'requirements-visualization.txt')
    manifest = {str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in bundle.rglob('*') if p.is_file() and p.name != 'sha256.json'}
    (bundle / 'sha256.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    archive = shutil.make_archive(str(bundle), 'zip', root_dir=bundle.parent, base_dir=bundle.name)
    print(f'\nTASK2 COMPLETE: {stats["mean"]:.2f} +/- {stats["std"]:.2f}%\n{archive}', flush=True)


if __name__ == '__main__':
    main()
