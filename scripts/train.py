import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    train_files, temp_files, _, _ = train_test_split(
        all_npy_files,
        stratify_labels,
        test_size=(config['val_ratio'] + config['test_ratio']),
        random_state=42,
    )

    val_test_split_ratio = config['test_ratio'] / (config['val_ratio'] + config['test_ratio'])
    val_files, test_files, _, _ = train_test_split(
        temp_files,
        [get_label_from_npy_filename(f) for f in temp_files], # Re-generate labels for temp_files
        test_size=val_test_split_ratio,
        random_state=42,
    )

    label_to_int = config['label_to_int']
    num_classes = len(label_to_int)

    train_dataset = VSLDataset_NPY(train_files, label_to_int)
    val_dataset = VSLDataset_NPY(val_files, label_to_int)
    # test_dataset is not used in train.py, but kept for consistency if needed later

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    model = Model(
        num_landmarks=TOTAL_LANDMARKS,
        landmark_dim=LANDMARK_DIM,
        hidden_size=config['hidden_units'],
        num_layers=config['lstm_layers'],
        num_classes=num_classes,
        num_heads=config['attention_heads']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

    best_val_accuracy = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(config['epochs']):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for i, (sequences, labels) in enumerate(train_pbar):
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': train_loss / (i + 1), 'acc': train_correct / train_total})

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
        with torch.no_grad():
            for i, (sequences, labels) in enumerate(val_pbar):
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'loss': val_loss / (i + 1), 'acc': val_correct / val_total})

        # --- Log Metrics and Print Summary ---
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{config['epochs']} -> Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"New best model saved with Val Acc: {best_val_accuracy:.4f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss'); ax1.set_xlabel('Epochs'); ax1.legend()
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy'); ax2.set_xlabel('Epochs'); ax2.legend()
    plt.savefig(os.path.join('experiments/runs', 'training_history.png'))
    plt.show()

if __name__ == '__main__':
    main()