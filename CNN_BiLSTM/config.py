from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


@dataclass
class TrainConfig:
    data_path: str = repo_path("data", "train", "rad_clean_merged.1.npy")
    noise_data_path: str = repo_path("data", "train", "data_noise_30002048_N.npy")
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    batch_size: int = 32
    noise_scale: float = 2.0
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 42

    input_size: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    output_size: int = 1
    dropout_rate: float = 0.2
    num_cnn_layers: int = 5
    base_channels: int = 32
    max_channels: int = 512

    learning_rate: float = 0.001
    num_epochs: int = 30
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    amp: bool = True
    device: str = "auto"
    log_dir: str = repo_path("CNN_BiLSTM", "logs")

    image_height: int = 32
    image_width: int = 64

    @property
    def sample_length(self) -> int:
        return self.image_height * self.image_width

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> "TrainConfig":
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must sum to 1.0, "
                f"got {ratio_sum:.6f}"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_cnn_layers <= 0:
            raise ValueError("num_cnn_layers must be positive")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.max_channels < self.base_channels:
            raise ValueError("max_channels must be >= base_channels")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height and image_width must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        return self

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        valid_fields = {field.name for field in fields(cls)}
        filtered = {
            key: value for key, value in values.items()
            if key in valid_fields and value is not None
        }
        return cls(**filtered).validate()


def build_train_arg_parser(
    description: str = "Training script for the CNN-BiLSTM microseismic denoising model",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=repo_path("data", "train", "rad_clean_merged.1.npy"),
        help="Path to the clean training waveform file (.npy)",
    )
    parser.add_argument(
        "--noise_data_path",
        type=str,
        default=repo_path("data", "train", "data_noise_30002048_N.npy"),
        help="Path to the noise waveform file (.npy)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training split ratio",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Testing split ratio",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=2.0,
        help="Scale factor applied to the noise waveforms",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes used by the dataloaders",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pinned host memory for CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting and model training",
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=1,
        help="Input feature dimension",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="BiLSTM hidden size",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of stacked BiLSTM layers",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=1,
        help="Output feature dimension",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="BiLSTM dropout rate",
    )
    parser.add_argument(
        "--num_cnn_layers",
        type=int,
        default=5,
        help="Number of 1D convolutional layers",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=32,
        help="Number of channels in the first convolutional layer",
    )
    parser.add_argument(
        "--max_channels",
        type=int,
        default=512,
        help="Upper bound used when expanding CNN channels",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.5,
        help="Factor applied by ReduceLROnPlateau",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=3,
        help="Epoch patience before reducing the learning rate",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Epoch patience before early stopping",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset early stopping",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AMP mixed-precision training when CUDA is available",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device specifier: auto, cpu, cuda, or cuda:0",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=repo_path("CNN_BiLSTM", "logs"),
        help="Directory used for training logs and checkpoints",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=32,
        help="Height used to reshape 1D waveforms for SSIM loss",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=64,
        help="Width used to reshape 1D waveforms for SSIM loss",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig.from_dict(vars(args))
