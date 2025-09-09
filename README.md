# VSL Recognition Project

This project implements a Visual Sign Language (VSL) recognition system using a Dual-Attention LSTM model. It processes gloss videos, extracts keypoints using MediaPipe, and trains a deep learning model to classify sign language gestures.

## Project Structure

```
project_name/
│── configs/                # Configuration files (YAML/JSON)
│   └── default.yaml        # Default configuration for data, model, and training
│
│── data/                   # Raw + processed data storage
│   ├── raw/                # Placeholder for raw video data (e.g., zipped files)
│   ├── processed/          # Placeholder for processed data (e.g., extracted keypoints)
│   └── datasets.py         # PyTorch Dataset classes for loading keypoint data
│
│── models/                 # Model definitions
│   ├── base_model.py       # (Optional) Base model or abstract classes
│   └── my_model.py         # Dual-Attention LSTM model implementation
│
│── utils/                  # Utility functions
│   ├── data_utils.py       # Functions for keypoint extraction, augmentation, and NPY dataset creation
│   ├── train_utils.py      # (Optional) Utilities for training loop (e.g., early stopping, learning rate schedulers)
│   ├── metrics.py          # (Optional) Custom metrics
│   ├── logger.py           # (Optional) Logging utilities
│   └── seed.py             # (Optional) Seeding for reproducibility
│
│── trainers/               # Training & evaluation loops
│   ├── trainer.py          # (Optional) Encapsulates training logic
│   └── evaluator.py        # (Optional) Encapsulates evaluation logic
│
│── scripts/                # Scripts for running experiments
│   ├── train.py            # Main script to train the VSL recognition model
│   ├── evaluate.py         # Script to evaluate the trained model on the test set
│   └── predict.py          # (Optional) Script for making predictions on new data
│
│── experiments/            # Save logs, checkpoints, results
│   ├── runs/               # Training logs, plots (e.g., loss/accuracy curves, confusion matrix)
│   └── checkpoints/         # Saved model checkpoints (e.g., best_model.pth)
│
│── requirements.txt        # Python dependencies
│── setup.py                # (Optional) Package setup
│── README.md               # Project documentation
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd vsl-recognition
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your data:**
    *   Place your zipped gloss video dataset (e.g., `vsl-vocab-demo.zip`) in a location accessible by the project (e.g., `/content/drive/MyDrive/VSL/` if using Google Colab, or `data/raw/` if local).
    *   Update the `base_data_folder` and `labels_file_path` in `configs/default.yaml` if your data is not in the expected `/content/vsl_dataset` path after extraction.

## Usage

### 1. Data Preprocessing

The `create_structured_npy_dataset` function in `utils/data_utils.py` handles video extraction, keypoint detection using MediaPipe, and saving them as standardized `.npy` files. This step is typically run once.

To run the data preprocessing:

```bash
python -c "from utils.data_utils import create_structured_npy_dataset; import yaml; config = yaml.safe_load(open('configs/default.yaml')); create_structured_npy_dataset(config)"
```
*Note: This command assumes you have already extracted your zipped dataset to the path specified in `configs/default.yaml` or that the script can find `data.txt` within `base_data_folder`.*

### 2. Training the Model

To train the model, run the `train.py` script:

```bash
python scripts/train.py
```

Training progress, including loss and accuracy curves, will be saved in the `experiments/runs/` directory, and the best model checkpoint will be saved in `experiments/checkpoints/`.

### 3. Evaluating the Model

To evaluate the trained model on the test set, run the `evaluate.py` script:

```bash
python scripts/evaluate.py
```

This will output the test accuracy, a classification report, and a confusion matrix, which will be saved in `experiments/runs/`.

## Configuration

All key parameters for data processing, model architecture, and training are managed in `configs/default.yaml`. You can modify this file to adjust hyperparameters or data paths.

## Dependencies

See `requirements.txt` for a list of all Python dependencies.# vsl-recognition
