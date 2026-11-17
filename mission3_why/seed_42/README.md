# seed 42

## 50k

- 关系测试准确率：**99.175%**
- 数字测试准确率：**98.45%**
- 最佳 epoch：4
- 错位（真实类→主预测类）：无，恒等映射
- Spearman ρ：0.8211
- Discovered order：2 → 1 → 0 → 3 → 4 → 7 → 6 → 8 → 9 → 5

文件：`metrics_50k.json`、`history_50k.csv`、`checkpoint_50k.pth`、`confusion_matrix_50k.csv`、`confusion_matrix_50k/`、`embedding_50k/`

## 500k

- 关系测试准确率：**99.365%**
- 数字测试准确率：**99.07%**
- 最佳 epoch：5
- 错位（真实类→主预测类）：无，恒等映射
- Spearman ρ：0.7826
- Discovered order：2 → 0 → 3 → 1 → 4 → 7 → 9 → 8 → 5 → 6

文件：`metrics_500k.json`、`history_500k.csv`、`checkpoint_500k.pth`、`confusion_matrix_500k.csv`、`confusion_matrix_500k/`、`embedding_500k/`

## 图

![50k t-SNE](embedding_50k/tsne.png)

![50k PCA 小提琴](embedding_50k/pca_violin.png)

![50k 混淆矩阵](confusion_matrix_50k/confusion_matrix_normalized.png)

![500k t-SNE](embedding_500k/tsne.png)

![500k PCA 小提琴](embedding_500k/pca_violin.png)

![500k 混淆矩阵](confusion_matrix_500k/confusion_matrix_normalized.png)
