import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
import os


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
    arr = np.asarray(signals)
    abs_arr = np.abs(arr)

    if arr.ndim == 1:
        max_val = abs_arr.max()
        if max_val == 0:
            return arr
        return arr / max_val

    elif arr.ndim == 2:
        max_vals = abs_arr.max(axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0
        return arr / max_vals



def load_text_data(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()
    data = [float(line.strip()) for line in data]
    return np.array(data)


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


class CSVFullSignalDataset(Dataset):
    """
    根据 CSV 的 file_path 列，逐条加载、归一化原始一维信号（不切分），
    返回：
      signal_tensor: torch.Tensor, shape (L, 1)
      file_path:      str
    """
    def __init__(self, csv_path: str, dataset_path: str):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.names = df['file_name'].tolist()
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        fp = os.path.join(self.dataset_path, name)
        sig = load_text_data(fp)
        sig = normalize_amplitude(sig)
        t = torch.tensor(sig, dtype=torch.float32)
        t = t.unsqueeze(-1)
        return t, name

def get_full_signal_loader(csv_path: str,
                           dataset_path: str,
                           batch_size: int = 1,
                           shuffle: bool = False,
                           num_workers: int = 0,
                           pin_memory: bool = True):
    """
    返回 DataLoader，每次迭代得到：
      signals: list of torch.Tensor, 每个 (L_i,1)
      paths:   list of str, 对应的 file_path
    之后在预测里，再按 chunk_size 切分：
      for sig, fp in zip(signals, paths):
          # sig.shape = (L,1)
          # 按 chunk_size pad/slice -> model 预测 -> 拼接
    """
    ds = CSVFullSignalDataset(csv_path, dataset_path)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            [item[1] for item in batch]
        ),
        persistent_workers=(num_workers > 0)
    )
    return loader
