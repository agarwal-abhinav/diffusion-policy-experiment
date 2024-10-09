import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class BinaryClassificationDataset(Dataset):
    def __init__(self, data_0_path, data_1_path):
        # Load data from npy files
        self.data_0 = np.load(data_0_path)
        self.data_1 = np.load(data_1_path)
        self.data_0 = self.data_0.reshape(self.data_0.shape[0], -1)
        self.data_1 = self.data_1.reshape(self.data_1.shape[0], -1)
        
        # Create labels
        self.labels_0 = np.zeros(len(self.data_0))
        self.labels_1 = np.ones(len(self.data_1))

        # Combine data and labels
        self.data = np.concatenate([self.data_0, self.data_1], axis=0)
        self.labels = np.concatenate([self.labels_0, self.labels_1], axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def get_dataloaders(data_0_path, data_1_path, batch_size, val_split=0.2):
    dataset = BinaryClassificationDataset(data_0_path, data_1_path)
    
    # Split dataset into training and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader