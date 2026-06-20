from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def normalize_amplitude(signals: np.ndarray) -> np.ndarray:
    arr = np.asarray(signals)
    abs_arr = np.abs(arr)

    if arr.ndim == 1:
        max_val = abs_arr.max()
        if max_val == 0:
            return arr
        return arr / max_val

    if arr.ndim == 2:
        max_vals = abs_arr.max(axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0
        return arr / max_vals

    raise ValueError(f"Expected a 1D or 2D array, got shape {arr.shape}")


def load_text_data(file_path: str) -> np.ndarray:
    with open(file_path, "r", encoding="utf-8") as file:
        data = [float(line.strip()) for line in file]
    return np.asarray(data, dtype=np.float32)


def load_training_arrays(
    clean_path: str,
    noise_path: str,
    noise_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    clean_np = np.load(clean_path, allow_pickle=False)
    noise_np = np.load(noise_path, allow_pickle=False)
    if clean_np.shape != noise_np.shape:
        raise ValueError(
            "Clean and noise arrays must share the same shape, "
            f"got {clean_np.shape} and {noise_np.shape}"
        )

    clean_np = normalize_amplitude(clean_np)
    noisy_np = clean_np + noise_np * noise_scale
    return noisy_np.astype(np.float32), clean_np.astype(np.float32)


def split_dataset_indices(
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    if num_samples <= 0:
        raise ValueError("Dataset is empty")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must sum to 1.0, "
            f"got {ratio_sum:.6f}"
        )

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(num_samples)

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    splits = {
        "train": shuffled_indices[:train_end],
        "val": shuffled_indices[train_end:val_end],
        "test": shuffled_indices[val_end:],
    }

    if min(len(indices) for indices in splits.values()) == 0:
        raise ValueError(
            "One of the dataset splits is empty. Adjust the split ratios or provide more data."
        )
    return splits


class SignalNoiseDataset(Dataset):
    def __init__(self, noisy_data: np.ndarray, clean_data: np.ndarray):
        super().__init__()
        if noisy_data.shape != clean_data.shape:
            raise ValueError(
                "Noisy and clean arrays must share the same shape, "
                f"got {noisy_data.shape} and {clean_data.shape}"
            )
        self.noisy = torch.from_numpy(noisy_data.astype(np.float32)).unsqueeze(-1)
        self.clean = torch.from_numpy(clean_data.astype(np.float32)).unsqueeze(-1)

    def __len__(self) -> int:
        return self.clean.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[idx], self.clean[idx]


def get_dataloaders(
    clean_path: str,
    noise_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    batch_size: int = 32,
    noise_scale: float = 2.0,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    noisy_np, clean_np = load_training_arrays(clean_path, noise_path, noise_scale=noise_scale)
    split_indices = split_dataset_indices(
        num_samples=clean_np.shape[0],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    datasets = {
        name: SignalNoiseDataset(noisy_np[index], clean_np[index])
        for name, index in split_indices.items()
    }

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


class CSVFullSignalDataset(Dataset):
    def __init__(self, csv_path: str, dataset_path: str):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.names = df["file_name"].tolist()
        self.dataset_path = dataset_path

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        name = self.names[idx]
        full_path = os.path.join(self.dataset_path, name)
        signal = normalize_amplitude(load_text_data(full_path))
        tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(-1)
        return tensor, name


def get_full_signal_loader(
    csv_path: str,
    dataset_path: str,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = CSVFullSignalDataset(csv_path, dataset_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            [item[1] for item in batch],
        ),
    )
