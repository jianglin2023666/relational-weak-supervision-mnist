# 任务2交付：全监督MNIST基线（fh）

本目录提交一次完整的 **100轮、batch64、五seed** 全监督实验及其分析。实验标识：`supervised_paper100_bs64_20261113_201811`。

数字测试准确率：**99.29% ± 0.08%**，即五seed均值99.292%、样本标准差0.0772658个百分点（ddof=1）。seed42的测试准确率为99.42%，Spearman rho为0.317265，PCA第一主成分解释23.09%的方差。

## 先看哪些文件

| 文件或目录 | 用途 |
|---|---|
| [分析报告.md](分析报告.md) | 正式分析，含训练动态、分类统计、混淆矩阵、错分样本和特征顺序解释 |
| [汇报提纲.md](汇报提纲.md) | 已填入共享PPT第2、3页的内容与讲稿 |
| [task5_handoff/README.md](task5_handoff/README.md) | 任务5取数说明、单位和公平比较规则 |
| [task5_handoff/supervised_summary.csv](task5_handoff/supervised_summary.csv) | 一行模型汇总，可直接进入统一对比表 |
| [training_results/runs.csv](training_results/runs.csv) | 五seed原始测试统计 |
| [training_results/summary.json](training_results/summary.json) | 均值、样本标准差及设备信息 |
| [analysis_data/analysis_summary.json](analysis_data/analysis_summary.json) | 详细派生分析指标 |
| `figures/` | 可插入报告或PPT的PNG，以及新增图形的可编辑SVG |
| `predictions/` | seed42全测试集预测和58张错分图片数据 |
| `training_results/seed_*/` | 最佳权重、100轮历史、混淆矩阵及各类准确率 |
| `configs/`、`scripts/` | 固定配置、训练入口及无需重训的分析复现工具 |
| `source_snapshot/` | 训练与模型等关键源文件快照，供审阅和版本记录 |
| [validation.json](validation.json)、`sha256.json` | 数值一致性核验记录与交付文件校验和 |

seed42预先固定用于图表分析，checkpoint由验证集准确率选择。该seed在本次测试最高并不是选图依据。embedding使用相同官方测试图片索引、每类200张，供其他方法对照。

## 训练配置

- MNIST：50000训练、10000验证、10000测试；像素除以255。
- BaseModel输出128维embedding，分类头Linear(128,10)，交叉熵。
- Adam，lr=0.001，weight_decay=0.0001，100 epochs，batch64。
- seeds=42、43、44、45、46；每seed78200次优化器更新。
- 按验证准确率选最佳checkpoint，同分保留较早轮次，完成训练后测试。
- 本机CPU、4线程、num_workers=0；详细依赖见requirements-lock.txt。

主要训练超参数对齐[原报告第4.2节](https://cs231n.stanford.edu/2025/papers/text_file_840591902-CS_231N_First_Draft.pdf)，分析参照其Figures 2、4、7。原报告自身的准确率记载不一致，且可视化采样与本课程规则不同，具体限制见分析报告。

## 复现分析与图表（不训练）

将 `mission2_fh` 文件夹放在项目根目录，在项目Python环境中执行：

```powershell
python -m pip install -r mission2_fh/requirements.txt -r mission2_fh/requirements-visualization.txt
python mission2_fh/scripts/analyze_saved_results.py
python mission2_fh/scripts/validate_delivery.py
```

第一条从保存的CSV、JSON、NPZ重建派生数据和新增科学图表，不重新训练、不重新选择checkpoint，也不重新运行t-SNE。t-SNE原始坐标和图片直接保存在 `figures/embedding_seed42/`。

如果重建操作改变了图形字节，校验和可能变化；重新封装交付材料时应更新sha256.json，不能忽略校验失败。

### 重新核对固定checkpoint

在原项目中准备好MNIST测试数据后执行：

```powershell
python mission2_fh/scripts/evaluate_seed42.py --project-root .
```

这会在CPU上重新加载交付的seed42权重，导出10000张测试预测，并验证其混淆矩阵和保存的embedding，不更新模型参数。

### 开展一次新的训练

以下命令会重新训练五seed各100轮，并将新结果写入原项目的 `training_results/supervised/` 时间戳目录：

```powershell
python mission2_fh/scripts/train.py --project-root .
```

无需为阅读本报告运行训练。此训练入口调用原项目训练代码，使用交付目录中的明确100轮配置，不修改共享默认配置。新训练产生的权重如要进一步绘图，可使用项目自带embedding_analysis和confusion_matrix_analysis命令。

源代码快照用于记录，不是独立可安装的完整项目。完整运行需要本项目源码、相应依赖和MNIST数据；本目录不附带虚拟环境或全部MNIST原始数据。

## 提交建议

将压缩包中的 `mission2_fh` 目录作为仓库顶层交付目录，PR标题与描述可使用 [PR说明.md](PR说明.md)。[团队共享PPT](https://docs.google.com/presentation/d/19JnjjdOZlTxPAnloK0wt_idIAteMixEWlT1_je9L8dE/edit#slide=id.h43316f9c7a0f8ff8_5_3)第2、3页已填入本次结果及演讲备注，保留第一页和团队模板。材料包准备不自动上传或合并PR。
