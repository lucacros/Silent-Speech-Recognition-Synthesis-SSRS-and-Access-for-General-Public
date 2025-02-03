import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils import pass_band_filter

class EMGDataset(Dataset):
    def __init__(self, data_folder):
        self.features = []
        self.labels = []
        self.class_labels = os.listdir(data_folder)  # subfolders names are the class labels
        self.label_counts = {label: 0 for label in self.class_labels}  # Initialize a dictionary to count samples per label

        for label, class_folder in enumerate(self.class_labels):
            class_path = os.path.join(data_folder, class_folder)
            for file_name in os.listdir(class_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(class_path, file_name)
                    audio, fs = librosa.load(file_path, sr=None, mono=False)  # mono=False --> stereo audio
                    # Verify that the audio is stereo
                    if audio.ndim == 2:
                        emg_filtered = pass_band_filter(audio[1,:])
                        self.features.append(audio[1, :])  # EMG signal
                        self.labels.append(label)
                        self.label_counts[class_folder] += 1  # Increment the count for the current label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class EMGDataModule(pl.LightningDataModule):
    def __init__(self, data_folder, batch_size=32, num_workers=4):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = EMGDataset(self.data_folder)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# Example usage
data_folder = 'DataBase/CHEEK'  # Replace with your actual data folder path
batch_size = 16
num_workers = 4

data_module = EMGDataModule(data_folder, batch_size, num_workers)
data_module.setup()

print("Sample number: ", len(data_module.dataset))
print("Data samples per label: ", data_module.dataset.label_counts)