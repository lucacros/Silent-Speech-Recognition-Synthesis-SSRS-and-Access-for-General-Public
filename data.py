import os
import numpy as np
import librosa
import torch as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchmetrics
import matplotlib.pyplot as plt
from utils import pass_band_filter, audio_to_mel_spectrogram

class CHEEKDataset(Dataset):
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
                        emg_filtered = pass_band_filter(audio[1,:]) # EMG signal
                        mel_signal = audio_to_mel_spectrogram(emg_filtered)
                        self.features.append(mel_signal)  
                        self.labels.append(label)
                        self.label_counts[class_folder] += 1  # Increment the count for the current label
        print("Data loaded successfully")    
        print(self.label_counts)         

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return nn.tensor(feature, dtype=nn.float32), nn.tensor(label, dtype=nn.long)

class CHEEKDataModule(pl.LightningDataModule):
    def __init__(self, data_folder,test_folder, batch_size=32, num_workers=4):
        super().__init__()
        self.data_folder = data_folder
        self.test_folder = test_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        
        if stage == "fit" or stage is None:
            cheek_dataset = CHEEKDataset(self.data_folder)
            train_size = int(0.8 * len(cheek_dataset))
            test_size = len(cheek_dataset) - train_size
            cheek_train, cheek_valid =  nn.utils.data.random_split(cheek_dataset, [train_size, test_size])
            self.cheek_train, self.cheek_valid = cheek_train, cheek_valid

        
        if stage == "test" or stage is None:

            self.cheek_test = CHEEKDataset(self.test_folder)

    def train_dataloader(self):
        return DataLoader(self.cheek_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cheek_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cheek_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class CHEEKClassifier(pl.LightningModule):
    def __init__(self, output_shape):
        super(CHEEKClassifier,self).__init__()
        
        self.output_shape = output_shape
        self.save_hyperparameters()
        self.learning_rate = 0.001
        # Model architecture, # INPUT [40,87]
        self.conv_layers = nn.nn.Sequential(
            nn.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # [1, 40, 87] -> [32, 40, 87]
            nn.nn.ReLU(),
            nn.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [32, 40, 87] -> [32, 20, 43]
            nn.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # [32, 20, 43] -> [64, 20, 43]
            nn.nn.ReLU(),
            nn.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [64, 10, 21] -> [64, 10, 21]
            nn.nn.Flatten(),
            nn.nn.Linear(64*10*21, 128),  # Ajustez en fonction de la sortie des couches convolutionnelles
            nn.nn.ReLU(),
            nn.nn.Linear(128, self.output_shape),
        )

        self.acc_train = torchmetrics.Accuracy(task='multiclass', num_classes=15)
        self.acc_test = torchmetrics.Accuracy(task='multiclass', num_classes=15)
        self.acc_valid = torchmetrics.Accuracy(task='multiclass', num_classes=15)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        

    

    def forward(self,x):
        x= x.unsqueeze(1)
        print("Input size:", x.size()) 
        x = self.conv_layers(x)
        print(x.size())
        return x

    def configure_optimizers(self):
        optimizer = nn.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch     
        estim = self.forward(x)
        loss = nn.nn.CrossEntropyLoss()(estim, y)
        preds = nn.argmax(estim, dim=1)
        acc = self.acc_train(preds, y)

        # Stocker la perte et la précision
        self.training_step_outputs.append(loss.item())

        self.log('train_acc', acc, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # TODO : Define your Validation Step
        # What is the difference between the Training and the Validation Step ?
        # The Validation step only verify the model accuracy without give in training 
        x,y = batch
        estim = self.forward(x)
        loss = nn.nn.CrossEntropyLoss()(estim,y)
        preds = nn.argmax(estim, dim=1)  # Get the predicted class labels
        acc = self.acc_valid(preds, y)
        # Don't remove the next line, you will understand why later
        self.log('val_acc', acc)
        self.log('val_loss', loss)

        self.validation_step_outputs.append(loss.item())


    def test_step(self, batch, batch_idx):
        x,y = batch
        estim = self.forward(x)
        loss = nn.nn.CrossEntropyLoss()(estim,y)
        preds = nn.argmax(estim, dim=1)  # We accumulate every accuracy
        acc = self.acc_test(preds, y)
        # Don't remove the next line, you will understand why later
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def test_epoch_start(self):
        self.accc= 0

    def on_test_epoch_end(self):
        self.acc = self.acc_test #torch metrics
        self.log('Final Accuracy', self.acc) #average test batch




# Example usage
data_folder = 'DataBase/CHEEK'  # Replace with your actual data folder path
test_folder = 'DataBase/CHEEK_test'
batch_size = 16
num_workers = 4


data_module = CHEEKDataModule(data_folder, test_folder, batch_size, num_workers)
model = CHEEKClassifier(15)
print(model)


trainer = pl.Trainer(devices='auto',max_epochs=20,accelerator='auto')
trainer.fit(model, data_module)
#nn.save(model, 'model.pth')
trainer.test(model,datamodule=data_module)

