# seed 44

## 50k

- 关系测试准确率：**97.950%**
- 数字测试准确率：**42.04%**
- 最佳 epoch：4
- 错位（真实类→主预测类）：3→4, 4→5, 5→6, 6→7, 7→8, 8→9
- Spearman ρ：0.6759
- Discovered order：2 → 0 → 1 → 3 → 7 → 8 → 4 → 9 → 6 → 5

文件：`metrics_50k.json`、`history_50k.csv`、`checkpoint_50k.pth`、`confusion_matrix_50k.csv`、`confusion_matrix_50k/`、`embedding_50k/`

## 500k

- 关系测试准确率：**98.245%**
- 数字测试准确率：**48.75%**
- 最佳 epoch：3
- 错位（真实类→主预测类）：3→4, 4→5, 5→6, 6→7, 7→8
- Spearman ρ：0.7418
- Discovered order：2 → 3 → 1 → 0 → 4 → 8 → 9 → 5 → 7 → 6

文件：`metrics_500k.json`、`history_500k.csv`、`checkpoint_500k.pth`、`confusion_matrix_500k.csv`、`confusion_matrix_500k/`、`embedding_500k/`

## 图

![50k t-SNE](embedding_50k/tsne.png)

![50k PCA 小提琴](embedding_50k/pca_violin.png)

![50k 混淆矩阵](confusion_matrix_50k/confusion_matrix_normalized.png)

![500k t-SNE](embedding_500k/tsne.png)

![500k PCA 小提琴](embedding_500k/pca_violin.png)

![500k 混淆矩阵](confusion_matrix_500k/confusion_matrix_normalized.png)
