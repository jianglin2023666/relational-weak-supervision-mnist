# mission3 — 任务三：基础弱监督模型（按 seed 归档）

结构化关系监督 `WeaklySupervisedModel`，50k / 500k 两组关系数据，seed 42–46。

## 汇总（每个 seed 一行）

| seed | 50k 关系 | 50k 数字 | 50k ρ | 50k 错位 | 500k 关系 | 500k 数字 | 500k ρ | 500k 错位 |
|---|---|---|---|---|---|---|---|---|
| 42 | 99.17 | 98.45 | 0.821 | 无 | 99.36 | 99.07 | 0.783 | 无 |
| 43 | 98.94 | 98.26 | 0.678 | 无 | 98.36 | 41.62 | 0.809 | 3→4, 4→5, 5→6, 6→7, 7→8, 8→9 |
| 44 | 97.95 | 42.04 | 0.676 | 3→4, 4→5, 5→6, 6→7, 7→8, 8→9 | 98.25 | 48.75 | 0.742 | 3→4, 4→5, 5→6, 6→7, 7→8 |
| 45 | 97.86 | 49.25 | 0.812 | 3→4, 4→5, 5→6, 6→7, 7→8 | 98.33 | 34.95 | 0.669 | 2→3, 3→4, 4→5, 5→6, 6→7, 7→8, 8→9 |
| 46 | 98.06 | 32.17 | 0.661 | 2→3, 3→4, 4→5, 5→6, 6→7, 7→8, 8→9 | 98.31 | 41.44 | 0.805 | 3→4, 4→5, 5→6, 6→7, 7→8, 8→9 |

- 50k：关系 98.40 ± 0.61%，数字 64.03 ± 31.91%
- 500k：关系 98.52 ± 0.47%，数字 53.17 ± 26.12%

低准确率的 seed 不是随机混淆，而是**保序但非单射的标号重排**：一段连续类别整体 +1 挪位（如 3→4, 4→5, …），两端的类不变。这种解只损失被合并那一对相邻类的关系对（约占 1.7–2.1%），所以关系准确率只掉约 1 个百分点，而数字准确率掉 50 个百分点以上。

补充证据：把 5 个 seed 的 embedding 全跑了（见上表 ρ 列），错位 seed 的 ρ 同样在 0.66–0.82，t-SNE 的 10 个簇同样分离清晰（对比 `seed_44/embedding_500k/tsne.png` 数字仅 48.75% 与 `seed_42/embedding_50k/tsne.png` 数字 98.45%）。**特征与顺序都学到了，错的只是簇到标号的映射**，所以 ρ 也不能用来预测数字分类准确率。

## 目录结构

```text
mission3/
├── README.md                        本文件（索引 + 汇总表）
├── 任务三_WeaklySupervisedModel.md    实验报告（图片链接已指向本目录内文件）
├── 任务三_WeaklySupervisedModel.pptx  2 页汇报 PPT
├── figures/
│   ├── structure.png                结构化关系概率计算图
│   ├── seed_summary.png             五 seed 关系/数字准确率对比
│   └── seed_summary.json
├── summary/
│   ├── runs_50k.csv, runs_500k.csv          每个 seed 的原始指标
│   ├── summary_50k.*, summary_500k.*        mean/min/max 统计
│   ├── config_*.json, metadata_*.json       训练配置与硬件信息
│   └── seed_accuracy_summary.csv            上面那张表的机器可读版
└── seed_42/ … seed_46/
    ├── metrics_50k.json, metrics_500k.json  单个 seed 的完整指标
    ├── history_50k.csv, history_500k.csv    每 epoch 训练/验证曲线
    ├── checkpoint_50k.pth, checkpoint_500k.pth
    ├── confusion_matrix_50k.csv, confusion_matrix_500k.csv
    ├── confusion_matrix_50k/  confusion_matrix_500k/   计数图 + 归一化图 + 准确率 JSON
    └── embedding_50k/  embedding_500k/     t-SNE、PCA 小提琴、ρ、Discovered order
```

每个 `seed_XX/README.md` 有该 seed 的一页小结。

## 说明

- 按课程规则，正式 embedding 分析只要求 `seed_42`；本目录把 5 个 seed 的全部 embedding 都跑了，用来对比“高准确率 seed”与“错位 seed”的簇结构，属于补充材料。
- `embedding_*/embeddings.npz` 只保留在 `seed_42`（复现 t-SNE 用），其余 seed 保留图片、指标和采样坐标 CSV。
- 复现命令见报告第 6 节；训练原始目录仍在 `repo/training_results/weakly_supervised/`。
