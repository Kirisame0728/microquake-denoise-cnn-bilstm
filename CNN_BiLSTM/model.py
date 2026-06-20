from __future__ import annotations

import torch
import torch.nn as nn


def build_cnn_channels(
    num_layers: int,
    base_channels: int = 32,
    max_channels: int = 512,
) -> list[int]:
    channels: list[int] = []
    current = base_channels
    for _ in range(num_layers):
        channels.append(min(current, max_channels))
        current = min(current * 2, max_channels)
    return channels


class LSTMCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_size = cfg.input_size
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.output_size = cfg.output_size
        self.dropout_rate = cfg.dropout_rate
        self.num_cnn_layers = getattr(cfg, "num_cnn_layers", 5)
        self.base_channels = getattr(cfg, "base_channels", 32)
        self.max_channels = getattr(cfg, "max_channels", 512)
        self.cnn_channels = build_cnn_channels(
            num_layers=self.num_cnn_layers,
            base_channels=self.base_channels,
            max_channels=self.max_channels,
        )

        self.cnn = self._build_cnn_backbone(self.input_size, self.cnn_channels)
        self.lstm = nn.LSTM(
            input_size=self.cnn_channels[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)

    @staticmethod
    def _build_cnn_backbone(input_channels: int, cnn_channels: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in cnn_channels:
            layers.extend(
                (
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            batch_size = x.size(0)
            return x.reshape(batch_size, -1, self.input_size)
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            return x.unsqueeze(-1)
        raise ValueError(
            "Expected model input with 2, 3, or 4 dimensions, "
            f"but received shape {tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(x)
        batch_size = x.size(0)

        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out)
