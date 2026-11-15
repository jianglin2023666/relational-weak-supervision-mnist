# Mission 4：SiameseModel

本目录汇总任务四 **50k、1-shot 和 3-shot、seed 42–46** 的完整实验材料。

## 已完成的 1-shot 结果

- 每类 1 张标注图片，关系训练集 50k 对，关系训练 5 epochs，少样本分类头训练 100 epochs。
- 五个 seed 的关系测试准确率：**98.943% ± 0.121%**。
- 五个 seed 的数字测试准确率：**96.132% ± 0.784%**。
- 五次运行的标准差使用样本标准差（ddof=1）。运行设备为 CPU。
- 原始逐 seed 统计、summary、训练历史和数字混淆矩阵 CSV 在 `1shot/`。
- seed 42 的 t-SNE、PCA 小提琴图、Spearman ρ、`Discovered order` 在 `1shot/embedding_seed42/`。
- seed 42 的计数和按真实类别归一化混淆矩阵图在 `1shot/confusion_seed42/`。
- Siamese 结构图在 `siamese_architecture.svg`。
- 1-shot 与 3-shot 的统一对比数据在 `task5_comparison.csv`，可交给任务五合并总表。

建议表述：关系测试准确率接近 99%，并且五个 seed 间波动很小；冻结关系预训练 backbone 后，1-shot 数字准确率约为 96.1%，仍有一定 seed 间变化。关系判断与绝对数字分类是不同任务，不能直接将两个准确率当作同一指标比较。

## 3-shot 实验结果

- 每类 3 张标注图片，其余设置与 1-shot 一致。
- 五个 seed 的关系测试准确率：**98.943% ± 0.121%**。
- 五个 seed 的数字测试准确率：**96.920% ± 0.131%**。
- 逐 seed 结果、训练历史和混淆矩阵 CSV 在 `3shot/training_results/`。
- seed 42 的 embedding 分析在 `3shot/embedding_seed42/`；混淆矩阵图在 `3shot/confusion_seed42/`。
- 完整的交叉分析见 `comparison_analysis.md`。

配置文件 `siamese_3shot.json` 保存可重复运行设置，命令为：

```bash
python -m src.training.train_siamese --config mission4_zdy/siamese_3shot.json
```

程序输出目录会自动写入 `training_results/siamese/`。本次完整原始运行目录为 `training_results/siamese/siamese_50k_3shot_20261115_184207`；归档副本保留了汇总表、每个 seed 的指标、训练历史和混淆矩阵，不包含模型权重。

```bash
python -m src.visualization.embedding_analysis \
  --model siamese \
  --checkpoint "training_results/siamese/<3-shot实验目录>/seed_42/best_siamese_relation_model.pth" \
  --output-dir "mission4_zdy/3shot/embedding_seed42"

python -m src.visualization.confusion_matrix_analysis \
  --input "training_results/siamese/<3-shot实验目录>/seed_42/digit_confusion_matrix.csv" \
  --output-dir "mission4_zdy/3shot/confusion_seed42" \
  --title "Siamese 3-shot - Seed 42"
```

## 结果解释时需要说明

1. Siamese 两条输入分支调用同一个 `BaseModel` 实例，共享权重；关系头接收两个 128 维 embedding 的拼接向量并预测 `A>B` 概率。
2. 关系预训练后复制 backbone 并冻结，只用每类 1 张或 3 张图片训练分类头。
3. 对比 1-shot 和 3-shot 时同时报告关系测试准确率、数字测试准确率及五 seed 的均值和标准差。增加 shot 数只增加少样本分类阶段的标签数。
4. t-SNE 用于观察簇形态；PCA 第一主成分、Spearman ρ 和 `Discovered order` 用于观察 embedding 是否呈数字顺序。它们是描述性分析，不能单独证明分类准确率变化的原因。
5. 1-shot 与 3-shot 的关系测试统计、Spearman ρ 和 `Discovered order` 完全相同，因为两阶段流程先执行相同关系预训练，再改变分类头可用的标签数；新增标注只影响分类头训练阶段。
6. Siamese 关系头直接从两张图片的 embedding 学习二元关系，不依赖 `0<1<...<9` 这样的类别顺序公式；因此在没有自然类别顺序的数据集上，可以把关系标签替换成“同类/不同类”或“相似/不相似”等目标。这个结构上的灵活性说明了迁移的可能性，本次 MNIST 实验本身并未验证其他数据集上的效果。

## 归档提示

任务四材料已按组内命名约定整理在 `mission4_zdy/`。目录包含一页结构、结果与结论汇总 PPTX：`SISI_module_mission4_one_slide.pptx`。
