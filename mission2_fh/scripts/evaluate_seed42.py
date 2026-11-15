"""Export all seed42 test predictions; no training or checkpoint selection."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.project_root.resolve()))
    from data.dataset import SupervisedMNISTDataset
    from src.models.supervised_model import SupervisedModel
    torch.set_num_threads(4)
    model = SupervisedModel()
    checkpoint = torch.load(root / 'training_results/seed_42/best_supervised_model.pth',
                            map_location='cpu', weights_only=True)
    assert checkpoint['seed'] == 42 and checkpoint['epoch'] == 66
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    dataset = SupervisedMNISTDataset(split='test')
    probabilities = []
    with torch.inference_mode():
        for images, _ in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0):
            probabilities.append(model(images).softmax(1).numpy())
    probabilities = np.concatenate(probabilities)
    labels = np.asarray(dataset.labels)
    predictions = probabilities.argmax(1)
    matrix = np.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    with (root / 'training_results/seed_42/digit_confusion_matrix.csv').open(encoding='utf-8-sig') as file:
        expected = np.array([[int(row[f'pred_{i}']) for i in range(10)] for row in csv.DictReader(file)])
    assert np.array_equal(matrix, expected), 'Reloaded checkpoint predictions differ from saved confusion matrix.'
    wrong = np.flatnonzero(labels != predictions)
    output = root / 'predictions'
    output.mkdir(exist_ok=True)
    with (output / 'seed42_test_predictions.csv').open('w', encoding='utf-8-sig', newline='') as file:
        fields = ['dataset_index', 'true_digit', 'predicted_digit', 'correct', 'predicted_probability'] + [f'p_{i}' for i in range(10)]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i in range(len(labels)):
            writer.writerow({'dataset_index': i, 'true_digit': int(labels[i]),
                             'predicted_digit': int(predictions[i]), 'correct': int(labels[i] == predictions[i]),
                             'predicted_probability': float(probabilities[i, predictions[i]]),
                             **{f'p_{j}': float(probabilities[i, j]) for j in range(10)}})
    np.savez_compressed(output / 'seed42_misclassified_images.npz',
                        dataset_indices=wrong, images=dataset.data[wrong].numpy(),
                        true_labels=labels[wrong], predicted_labels=predictions[wrong],
                        predicted_probabilities=probabilities[wrong, predictions[wrong]])
    # Confirm the saved embedding sample indices/labels against the same official test split.
    saved = np.load(root / 'figures/embedding_seed42/embeddings.npz')
    indices = saved['selected_indices']
    assert np.array_equal(labels[indices], saved['labels'])
    extracted = []
    with torch.inference_mode():
        for start in range(0, len(indices), 256):
            images = dataset.data[indices[start:start + 256]].float() / 255.0
            extracted.append(model.get_embedding(images).numpy())
    extracted = np.concatenate(extracted)
    max_difference = float(np.abs(extracted - saved['embeddings']).max())
    assert np.allclose(extracted, saved['embeddings'], atol=1e-5, rtol=1e-5)
    result = {'checkpoint_reload_confusion_match': True, 'test_samples': len(labels),
              'correct': int((labels == predictions).sum()), 'errors': len(wrong),
              'embedding_max_absolute_difference': max_difference}
    (root / 'analysis_data/checkpoint_recheck.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
