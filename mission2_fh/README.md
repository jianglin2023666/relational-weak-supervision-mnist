# 任务2：全监督 MNIST 基线（fh）

五个 seed（42–46）各训练5轮，数字测试准确率 **98.98% ± 0.16%**，标准差使用 ddof=1。

| Seed | 最佳 epoch | 测试准确率 |
|---|---:|---:|
| 42 | 4 | 98.85% |
| 43 | 4 | 98.78% |
| 44 | 4 | 99.07% |
| 45 | 5 | 99.14% |
| 46 | 5 | 99.08% |

## 材料索引
- [实验报告](实验报告.md)：实验设置、解释及PPT建议。
- [训练结构图](figures/supervised_training_flow.png)。
- [五seed结果图](figures/five_seed_results.png)。
- [混淆矩阵](figures/confusion_matrix_seed42/confusion_matrix_normalized.png)。
- [t-SNE](figures/embedding_seed42/tsne.png)、[PCA小提琴图](figures/embedding_seed42/pca_violin.png)。
- [五seed原始统计](training_results/runs.csv)、[均值和标准差](training_results/summary.json)。
- [embedding指标](figures/embedding_seed42/embedding_metrics.json)：Spearman ρ约0.2392，Discovered order为6,0,5,4,8,1,2,9,3,7。
- training_results/seed_*/：最佳权重、训练历史、每类准确率和混淆矩阵CSV。

## 公平比较设置
训练/验证/测试为50000/10000/10000张图像，使用全部训练数字标签。
每个seed按验证集准确率选最佳模型。embedding固定seed42、每类200张测试图片，共2000张。
50k/500k关系对设置属于关系监督任务，任务2不重复同一全监督基线。

## 给任务5的数据
请使用 training_results 下的 runs.csv、summary.json、config.json、metadata.json，以及 embedding_seed42 下的JSON/CSV，准确率单位为百分数。

## 复现
在项目根目录创建 .venv 并安装本目录 requirements.txt 和 requirements-visualization.txt，然后执行：

```powershell
.\.venv\Scripts\python.exe -u mission2_fh/run_task2.py
```

此命令重新运行训练并输出新实验。仅查看已有结果无需运行。
run_task2.cmd使用Windows虚拟环境启动同一流程。原项目模型和训练代码保持不变。
已验证CPU环境的准确版本见requirements-lock.txt。共享PPT尚未编辑，PPT页面不包含在本PR中。
