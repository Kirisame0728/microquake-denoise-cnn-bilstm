import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Config:
    data_path        = "../data/train/rad_clean_merged.1.npy"
    noise_data_path  = "../data/train/data_noise_30002048_N.npy"
    train_ratio      = 0.8
    batch_size       = 32

    input_size       = 1
    hidden_size      = 64
    num_layers       = 2
    output_size      = 1
    dropout_rate     = 0.1

    learning_rate    = 0.001
    num_epochs       = 30
    scheduler_factor = 0.5
    scheduler_patience = 3


def normalize_amplitude(signals: np.ndarray) -> np.ndarray:
    max_vals = np.max(np.abs(signals), axis=1, keepdims=True)
    return signals / max_vals


class SignalNoiseDataset(Dataset):
    def __init__(self,
                 clean_path: str,
                 noise_path: str,
                 train: bool = True,
                 train_ratio: float = 0.8,
                 noise_scale: float = 2.0):
        super().__init__()
        clean_np = np.load(clean_path)
        noise_np = np.load(noise_path)
        assert clean_np.shape == noise_np.shape, "clean 与 noise 大小不一致"

        clean_np = normalize_amplitude(clean_np)
        noisy_np = clean_np + noise_np * noise_scale

        N = clean_np.shape[0]
        split = int(train_ratio * N)
        if train:
            clean_np = clean_np[:split]
            noisy_np = noisy_np[:split]
        else:
            clean_np = clean_np[split:]
            noisy_np = noisy_np[split:]

        self.clean = torch.from_numpy(clean_np.astype(np.float32)).unsqueeze(-1)
        self.noisy = torch.from_numpy(noisy_np.astype(np.float32)).unsqueeze(-1)

    def __len__(self):
        return self.clean.size(0)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


def get_dataloaders(clean_path: str,
                    noise_path: str,
                    train_ratio: float = 0.8,
                    batch_size: int = 32,
                    noise_scale: float = 2.0,
                    num_workers: int = 0,
                    pin_memory: bool = True):

    train_ds = SignalNoiseDataset(
        clean_path=clean_path,
        noise_path=noise_path,
        train=True,
        train_ratio=train_ratio,
        noise_scale=noise_scale)
    val_ds = SignalNoiseDataset(
        clean_path=clean_path,
        noise_path=noise_path,
        train=False,
        train_ratio=train_ratio,
        noise_scale=noise_scale)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader
