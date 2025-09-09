import os
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data.datasets import VSLDataset_NPY
from models.model import Model
from utils.data_utils import get_label_from_npy_filename, TOTAL_LANDMARKS, LANDMARK_DIM

def main():
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_npy_files = [os.path.join(config['output_npy_folder'], f) for f in os.listdir(config['output_npy_folder']) if f.endswith('.npy')]
    stratify_labels = [get_label_from_npy_filename(f) for f in all_npy_files]

    # Re-split data to ensure consistency with training script
    train_files, temp_files, _, _ = train_test_split(
        all_npy_files,
        stratify_labels,
        test_size=(config['val_ratio'] + config['test_ratio']),
        random_state=42,
    )

    val_test_split_ratio = config['test_ratio'] / (config['val_ratio'] + config['test_ratio'])
    _, test_files, _, _ = train_test_split(
        temp_files,
        [get_label_from_npy_filename(f) for f in temp_files],
        test_size=val_test_split_ratio,
        random_state=42,
    )

    label_to_int = config['label_to_int']
    int_to_label = {i: label for label, i in label_to_int.items()}
    num_classes = len(label_to_int)
    class_names = list(label_to_int.keys())

    test_dataset = VSLDataset_NPY(test_files, label_to_int)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = Model(
        num_landmarks=TOTAL_LANDMARKS,
        landmark_dim=LANDMARK_DIM,
        hidden_size=config['hidden_units'],
        num_layers=config['lstm_layers'],
        num_classes=num_classes,
        num_heads=config['attention_heads']
    )
    
    model_path = config['model_save_path']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    detailed_results = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for pred_idx, true_idx in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                detailed_results.append((pred_idx, true_idx))

    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report on Test Set:")
    print(classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    ))

    print("\n--- Detailed Prediction Analysis ---")
    wrong_predictions_count = 0
    print("\nWrong Predictions:")
    if not detailed_results:
        print("  No test samples processed.")
    for i, (predicted_idx, true_idx) in enumerate(detailed_results):
        predicted_word = int_to_label[predicted_idx]
        true_word = int_to_label[true_idx]
        if predicted_idx != true_idx:
            wrong_predictions_count += 1
            print(f"  Sample {i+1}: Predicted '{predicted_word}', Actual '{true_word}'")
    if wrong_predictions_count == 0:
        print("  No wrong predictions! Model achieved 100% accuracy on the test set.")
    print(f"\nTotal Wrong Predictions: {wrong_predictions_count}")
    print(f"Total Correct Predictions: {len(detailed_results) - wrong_predictions_count}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join('experiments/runs', 'confusion_matrix.png'))
    plt.show()

if __name__ == '__main__':
    main()