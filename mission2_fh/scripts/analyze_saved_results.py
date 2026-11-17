"""Rebuild numeric analyses and scientific plots from saved results only."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def figure_save(figure, stem):
    figure.savefig(ROOT / 'figures' / f'{stem}.png', dpi=220, bbox_inches='tight')
    figure.savefig(ROOT / 'figures' / f'{stem}.svg', bbox_inches='tight')
    plt.close(figure)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    analysis, handoff = ROOT / 'analysis_data', ROOT / 'task5_handoff'
    analysis.mkdir(exist_ok=True)
    handoff.mkdir(exist_ok=True)
    runs = read_csv(ROOT / 'training_results/runs.csv')
    histories = {s: read_csv(ROOT / f'training_results/seed_{s}/history.csv') for s in SEEDS}
    matrices, classes, errors, detailed = {}, [], [], []
    colors = plt.colormaps['tab10'](np.arange(5))
    for run in runs:
        seed = int(run['seed'])
        matrix = np.array([[int(row[f'pred_{j}']) for j in range(10)]
                           for row in read_csv(ROOT / f'training_results/seed_{seed}/digit_confusion_matrix.csv')])
        matrices[seed] = matrix
        total, correct = int(matrix.sum()), int(np.trace(matrix))
        assert total == 10000
        assert np.isclose(100 * correct / total, float(run['digit_test_accuracy']))
        history = histories[seed]
        assert len(history) == 100
        best_epoch = int(run['best_epoch'])
        val = np.array([float(r['val_digit_accuracy']) for r in history])
        assert best_epoch == int(np.argmax(val)) + 1
        assert np.isclose(float(run['best_val_digit_accuracy']), val.max())
        best = history[best_epoch - 1]
        detailed.append({'seed': seed, 'epochs': 100, 'batch_size': 64, 'optimizer_updates': 78200,
                         'best_epoch': best_epoch, 'best_val_digit_accuracy_percent': float(run['best_val_digit_accuracy']),
                         'digit_test_accuracy_percent': float(run['digit_test_accuracy']),
                         'digit_test_loss': float(run['digit_test_loss']), 'test_errors': total - correct,
                         'train_accuracy_at_best_epoch_percent': float(best['train_digit_accuracy']),
                         'train_loss_at_best_epoch': float(best['train_loss']), 'val_loss_at_best_epoch': float(best['val_loss']),
                         'epoch100_train_accuracy_percent': float(history[-1]['train_digit_accuracy']),
                         'epoch100_val_accuracy_percent': float(history[-1]['val_digit_accuracy']),
                         'epoch100_train_loss': float(history[-1]['train_loss']),
                         'epoch100_val_loss': float(history[-1]['val_loss']),
                         'min_val_loss': min(float(r['val_loss']) for r in history),
                         'first_epoch_val_accuracy_ge99': next((int(r['epoch']) for r in history if float(r['val_digit_accuracy']) >= 99), None),
                         'training_time_seconds': float(run['training_time_seconds']),
                         'checkpoint': f'training_results/seed_{seed}/best_supervised_model.pth'})
        for digit in range(10):
            tp = int(matrix[digit, digit]); support = int(matrix[digit].sum()); predicted = int(matrix[:, digit].sum())
            precision = tp / predicted if predicted else 0.0
            recall = tp / support
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            classes.append({'seed': seed, 'digit': digit, 'support': support, 'correct': tp,
                            'errors': support - tp, 'precision_percent': 100 * precision,
                            'recall_percent': 100 * recall, 'f1_percent': 100 * f1})
            for prediction in range(10):
                if prediction != digit and matrix[digit, prediction]:
                    errors.append({'seed': seed, 'true_digit': digit, 'predicted_digit': prediction,
                                   'count': int(matrix[digit, prediction]),
                                   'true_class_percent': 100 * matrix[digit, prediction] / support})
    errors.sort(key=lambda r: (r['seed'], -r['count'], r['true_digit'], r['predicted_digit']))
    values = np.array([float(r['digit_test_accuracy']) for r in runs])
    embedding = json.loads((ROOT / 'figures/embedding_seed42/embedding_metrics.json').read_text())
    saved = np.load(ROOT / 'figures/embedding_seed42/embeddings.npz')
    assert len(saved['labels']) == 2000 and all(sum(saved['labels'] == i) == 200 for i in range(10))
    pca = PCA(n_components=1)
    raw_projection = pca.fit_transform(saved['embeddings']).ravel()
    raw_rho = float(spearmanr(raw_projection, saved['labels']).statistic)
    oriented_projection = raw_projection * (-1 if raw_rho < 0 else 1)
    assert np.isclose(abs(raw_rho), embedding['spearman_rho'], atol=1e-6)
    assert np.allclose(oriented_projection, saved['pca_projection'], atol=1e-4)
    projection_order = sorted(range(10), key=lambda d: oriented_projection[saved['labels'] == d].mean())
    assert projection_order == embedding['discovered_order']
    summary = {'experiment_id': 'supervised_paper100_bs64_20261113_201811',
               'mean_digit_test_accuracy_percent': float(values.mean()),
               'std_digit_test_accuracy_percentage_points': float(values.std(ddof=1)),
               'std_ddof': 1, 'min_digit_test_accuracy_percent': float(values.min()),
               'max_digit_test_accuracy_percent': float(values.max()),
               'seeds': SEEDS, 'seed42_test_correct': int(np.trace(matrices[42])),
               'seed42_test_errors': 10000 - int(np.trace(matrices[42])),
               'seed42_top_confusions': [r for r in errors if r['seed'] == 42][:10],
               'seed42_class_metrics': [r for r in classes if r['seed'] == 42],
               'embedding_seed': 42, 'embedding_samples': 2000, 'embedding_samples_per_class': 200,
               'raw_spearman_rho': embedding['raw_spearman_rho'], 'spearman_rho': embedding['spearman_rho'],
               'spearman_p_value': embedding['spearman_p_value'],
               'pca_pc1_explained_variance_ratio': embedding['pca_explained_variance_ratio'],
               'discovered_order': embedding['discovered_order'], 'per_seed': detailed}
    save_json(analysis / 'analysis_summary.json', summary)
    save_csv(analysis / 'per_seed_results.csv', detailed)
    save_csv(analysis / 'per_class_metrics_all_seeds.csv', classes)
    save_csv(analysis / 'confusion_pairs_all_seeds.csv', errors)
    long_history = [dict(seed=s, **{k: v for k, v in row.items() if k != 'seed'})
                    for s in SEEDS for row in histories[s]]
    save_csv(analysis / 'training_history_all_seeds.csv', long_history)
    class_summary = []
    for digit in range(10):
        rows = [r for r in classes if r['digit'] == digit]
        class_summary.append({'digit': digit, 'recall_mean_percent': float(np.mean([r['recall_percent'] for r in rows])),
                              'recall_std_percentage_points': float(np.std([r['recall_percent'] for r in rows], ddof=1)),
                              'precision_mean_percent': float(np.mean([r['precision_percent'] for r in rows])),
                              'f1_mean_percent': float(np.mean([r['f1_percent'] for r in rows]))})
    save_csv(analysis / 'per_class_summary.csv', class_summary)
    handoff_summary = {'model': 'supervised', 'experiment_id': summary['experiment_id'],
                       'train_image_count': 50000, 'absolute_training_label_count': 50000,
                       'relation_train_pairs': '', 'shots_per_class': '', 'seed_count': 5,
                       'epochs_per_seed': 100, 'batch_size': 64, 'optimizer_updates_per_seed': 78200,
                       'digit_test_accuracy_mean_percent': float(values.mean()),
                       'digit_test_accuracy_std_percentage_points': float(values.std(ddof=1)),
                       'relation_test_accuracy_percent': '', 'std_ddof': 1,
                       'embedding_seed': 42, 'embedding_samples_per_class': 200,
                       'spearman_rho_seed42': embedding['spearman_rho'],
                       'pc1_explained_variance_ratio_seed42': embedding['pca_explained_variance_ratio'],
                       'discovered_order_seed42': ' '.join(map(str, embedding['discovered_order']))}
    save_csv(handoff / 'supervised_summary.csv', [handoff_summary])
    save_csv(handoff / 'supervised_per_seed.csv', detailed)
    save_csv(handoff / 'per_class_metrics.csv', classes)
    samples = read_csv(ROOT / 'figures/embedding_seed42/embedding_samples.csv')
    save_csv(handoff / 'embedding_sample_indices_seed42.csv',
             [{'dataset_index': r['dataset_index'], 'digit': r['digit']} for r in samples])
    save_json(handoff / 'embedding_metrics_seed42.json', embedding)
    # Paper-style training loss / accuracy curves, using the fixed seed42.
    h = histories[42]; epochs = np.arange(1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(epochs, [float(r['train_loss']) for r in h], label='Train')
    axes[0].plot(epochs, [float(r['val_loss']) for r in h], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Seed 42: training dynamics')
    axes[1].plot(epochs, [float(r['train_digit_accuracy']) for r in h], label='Train')
    axes[1].plot(epochs, [float(r['val_digit_accuracy']) for r in h], label='Validation')
    axes[1].axvline(66, color='#555555', ls='--', lw=1.2, label='Selected epoch 66')
    axes[1].set(xlabel='Epoch', ylabel='Digit accuracy (%)', title='Selection uses validation accuracy')
    for ax in axes: ax.legend(); ax.grid(alpha=.18)
    fig.tight_layout(); figure_save(fig, 'training_dynamics_seed42')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for s, color in zip(SEEDS, colors):
        h = histories[s]
        axes[0].plot(epochs, [float(r['val_digit_accuracy']) for r in h], color=color, label=f'Seed {s}', alpha=.85)
        axes[1].plot(epochs, [float(r['val_loss']) for r in h], color=color, label=f'Seed {s}', alpha=.85)
    axes[0].set(xlabel='Epoch', ylabel='Validation accuracy (%)', title='Validation accuracy across five seeds')
    axes[1].set(xlabel='Epoch', ylabel='Validation cross-entropy', title='Validation loss across five seeds')
    for ax in axes: ax.legend(fontsize=9); ax.grid(alpha=.18)
    fig.tight_layout(); figure_save(fig, 'training_dynamics_all_seeds')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(np.arange(5), values, s=70, c=colors, zorder=3)
    for i, v in enumerate(values): ax.annotate(f'{v:.2f}%', (i, v), xytext=(0, 10), textcoords='offset points', ha='center')
    ax.errorbar(5.5, values.mean(), yerr=values.std(ddof=1), fmt='D', color='#202020', capsize=7, label='Mean +/- sample SD')
    ax.set(xticks=[0, 1, 2, 3, 4, 5.5], xticklabels=['42', '43', '44', '45', '46', 'Mean'],
           ylabel='Digit test accuracy (%)', xlabel='Seed', ylim=(99.05, 99.6), title='Five independent runs: 99.292% +/- 0.077 pp')
    ax.grid(axis='y', alpha=.2); ax.legend(loc='upper right'); fig.tight_layout(); figure_save(fig, 'five_seed_accuracy')
    fig, ax = plt.subplots(figsize=(9, 4.8))
    means = [r['recall_mean_percent'] for r in class_summary]
    stds = [r['recall_std_percentage_points'] for r in class_summary]
    ax.errorbar(range(10), means, yerr=stds, fmt='o', capsize=5, color='#2765a3', label='Five-seed mean +/- sample SD')
    ax.set(xticks=range(10), xlabel='True digit', ylabel='Recall (%)', ylim=(97.8, 100.1), title='Per-class recall across five seeds')
    ax.legend(); ax.grid(axis='y', alpha=.2); fig.tight_layout(); figure_save(fig, 'per_class_recall')
    # Architecture overview with explicit inference output and supervised target.
    fig, ax = plt.subplots(figsize=(13, 3.4)); ax.set(xlim=(0, 13), ylim=(0, 3.4)); ax.axis('off')
    boxes = [(0.1, 'MNIST image\n[B,1,28,28]'), (3.3, 'Shared BaseModel\n128-D embedding'),
             (6.5, 'Linear(128,10)\n10 logits'), (9.7, 'Prediction\nargmax: digit 0-9')]
    for x, text in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.65), 2.65, 1.1, boxstyle='round,pad=0.08', facecolor='#e6eff8', edgecolor='#2d5f91', lw=1.5))
        ax.text(x + 1.325, 2.2, text, ha='center', va='center', fontsize=12)
    for x in [2.8, 6.0, 9.2]: ax.annotate('', xy=(x + .38, 2.2), xytext=(x, 2.2), arrowprops={'arrowstyle': '->', 'lw': 1.6})
    ax.text(7.82, .9, 'Cross-entropy(logits, true label)', ha='center', fontsize=12)
    ax.annotate('', xy=(7.82, 1.2), xytext=(7.82, 1.6), arrowprops={'arrowstyle': '->', 'lw': 1.6})
    ax.text(6.5, .25, 'Training updates both BaseModel and classifier; evaluation disables dropout.', ha='center', fontsize=11)
    fig.tight_layout(); figure_save(fig, 'supervised_training_flow')
    data = np.load(ROOT / 'predictions/seed42_misclassified_images.npz')
    count = min(24, len(data['dataset_indices']))
    fig, axes = plt.subplots(4, 6, figsize=(11, 8))
    for i, ax in enumerate(axes.ravel()):
        ax.axis('off')
        if i >= count: continue
        ax.imshow(np.squeeze(data['images'][i]), cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"#{int(data['dataset_indices'][i])}: {int(data['true_labels'][i])} to {int(data['predicted_labels'][i])}\np={data['predicted_probabilities'][i]:.2f}", fontsize=10)
    fig.suptitle('Seed 42: first 24 errors in test-index order (true to predicted)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95)); figure_save(fig, 'seed42_error_examples')
    print(json.dumps({'mean': values.mean(), 'std': values.std(ddof=1), 'seed42_errors': summary['seed42_test_errors'],
                      'rho': embedding['spearman_rho'], 'top_confusions': summary['seed42_top_confusions'][:5],
                      'per_seed_dynamics': detailed}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
