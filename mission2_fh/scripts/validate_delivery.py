"""Validate the delivery; --record writes numeric checks before packaging."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/supervised_paper100.json').read_text())
    assert config['epochs'] == 100 and config['batch_size'] == 64
    assert config['seeds'] == [42, 43, 44, 45, 46]
    run_rows = rows(ROOT / 'training_results/runs.csv')
    assert [int(r['seed']) for r in run_rows] == config['seeds']
    values = []
    for run in run_rows:
        seed = int(run['seed']); directory = ROOT / f'training_results/seed_{seed}'
        history = rows(directory / 'history.csv')
        assert [int(r['epoch']) for r in history] == list(range(1, 101))
        val = [float(r['val_digit_accuracy']) for r in history]
        assert int(run['best_epoch']) == int(np.argmax(val)) + 1
        matrix = np.array([[int(r[f'pred_{j}']) for j in range(10)] for r in rows(directory / 'digit_confusion_matrix.csv')])
        assert matrix.shape == (10, 10) and matrix.sum() == 10000
        accuracy = 100 * matrix.trace() / matrix.sum()
        assert np.isclose(accuracy, float(run['digit_test_accuracy']))
        values.append(accuracy)
        checkpoint = torch.load(directory / 'best_supervised_model.pth', map_location='cpu', weights_only=True)
        assert checkpoint['seed'] == seed and checkpoint['epoch'] == int(run['best_epoch'])
        assert checkpoint['config'] == config
        assert np.isclose(checkpoint['val_accuracy'], float(run['best_val_digit_accuracy']))
    summary = json.loads((ROOT / 'training_results/summary.json').read_text())['metrics']['digit_test_accuracy']
    assert np.isclose(np.mean(values), summary['mean'])
    assert np.isclose(np.std(values, ddof=1), summary['std'])
    samples = np.load(ROOT / 'figures/embedding_seed42/embeddings.npz')
    assert samples['embeddings'].shape == (2000, 128)
    assert len(np.unique(samples['selected_indices'])) == 2000
    assert all(sum(samples['labels'] == i) == 200 for i in range(10))
    embedding = json.loads((ROOT / 'figures/embedding_seed42/embedding_metrics.json').read_text())
    assert np.isclose(spearmanr(samples['pca_projection'], samples['labels']).statistic, embedding['spearman_rho'])
    assert sorted(range(10), key=lambda d: samples['pca_projection'][samples['labels'] == d].mean()) == embedding['discovered_order']
    predictions = rows(ROOT / 'predictions/seed42_test_predictions.csv')
    assert len(predictions) == 10000
    assert [int(r['dataset_index']) for r in predictions] == list(range(10000))
    targets = np.array([int(r['true_digit']) for r in predictions])
    pred = np.array([int(r['predicted_digit']) for r in predictions])
    probs = np.array([[float(r[f'p_{j}']) for j in range(10)] for r in predictions])
    assert np.allclose(probs.sum(1), 1, atol=1e-6) and np.array_equal(probs.argmax(1), pred)
    assert int((targets == pred).sum()) == 9942
    assert np.array_equal(targets[samples['selected_indices']], samples['labels'])
    wrong = np.load(ROOT / 'predictions/seed42_misclassified_images.npz')
    assert np.array_equal(wrong['dataset_indices'], np.flatnonzero(targets != pred))
    assert len(rows(ROOT / 'task5_handoff/supervised_per_seed.csv')) == 5
    assert len(rows(ROOT / 'task5_handoff/per_class_metrics.csv')) == 50
    assert len(rows(ROOT / 'task5_handoff/embedding_sample_indices_seed42.csv')) == 2000
    handoff = rows(ROOT / 'task5_handoff/supervised_summary.csv')[0]
    assert np.isclose(float(handoff['digit_test_accuracy_mean_percent']), np.mean(values))
    assert handoff['relation_test_accuracy_percent'] == ''
    for path in (ROOT / 'scripts').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
    for path in ROOT.rglob('*.md'):
        text = path.read_text(encoding='utf-8')
        assert len(re.findall(r'^```', text, flags=re.M)) % 2 == 0, path
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if args.record and target == 'validation.json':
                continue
            if not target.startswith(('https://', 'http://', '#')):
                assert (path.parent / target).is_file(), (path.name, target)
    recheck = json.loads((ROOT / 'analysis_data/checkpoint_recheck.json').read_text())
    assert recheck['checkpoint_reload_confusion_match'] and recheck['embedding_max_absolute_difference'] == 0.0
    result = {'status': 'passed', 'training_config': '100 epochs, batch64, seeds42-46',
              'history_rows': 500, 'test_samples_per_seed': 10000,
              'best_checkpoint_selection': 'first maximum validation accuracy',
              'mean_digit_test_accuracy_percent': float(np.mean(values)),
              'sample_std_percentage_points': float(np.std(values, ddof=1)),
              'seed42_reloaded_checkpoint_confusion_match': True,
              'seed42_embedding_max_absolute_difference': 0.0,
              'test_prediction_rows': 10000, 'seed42_errors': 58,
              'embedding_samples': 2000, 'embedding_samples_per_class': 200,
              'local_document_links_and_python_syntax': 'passed'}
    if args.record:
        (ROOT / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    else:
        manifest = json.loads((ROOT / 'sha256.json').read_text(encoding='utf-8'))
        files = {str(p.relative_to(ROOT)).replace('\\', '/') for p in ROOT.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.name != 'sha256.json'}
        assert files == set(manifest), 'Manifest file list differs from delivery files.'
        for path, expected in manifest.items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
        result['verified_file_hashes'] = len(manifest)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
