import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data_utils import get_label_from_npy_filename

class VSLDataset_NPY(Dataset):
    def __init__(self, file_paths, label_map):
        self.file_paths = file_paths
        self.label_map = label_map

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        sequence = np.load(file_path)
        label_name = get_label_from_npy_filename(file_path)
        label = self.label_map[label_name]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)