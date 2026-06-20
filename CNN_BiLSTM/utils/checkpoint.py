from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from CNN_BiLSTM.config import TrainConfig
except ImportError:
    from config import TrainConfig


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _extract_model_config(model: nn.Module) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for attr in (
        "input_size",
        "hidden_size",
        "num_layers",
        "output_size",
        "dropout_rate",
        "num_cnn_layers",
        "base_channels",
        "max_channels",
    ):
        if hasattr(model, attr):
            config[attr] = getattr(model, attr)
    return config


def _infer_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}

    conv_entries: list[tuple[int, str, torch.Tensor]] = []
    for key, tensor in state_dict.items():
        if key.startswith("cnn.") and key.endswith("weight") and tensor.ndim == 3:
            index = int(key.split(".")[1])
            conv_entries.append((index, key, tensor))
    conv_entries.sort(key=lambda item: item[0])

    if conv_entries:
        channels = [tensor.shape[0] for _, _, tensor in conv_entries]
        inferred["input_size"] = conv_entries[0][2].shape[1]
        inferred["num_cnn_layers"] = len(conv_entries)
        inferred["base_channels"] = channels[0]
        inferred["max_channels"] = max(channels)

    if "lstm.weight_hh_l0" in state_dict:
        inferred["hidden_size"] = state_dict["lstm.weight_hh_l0"].shape[1]

    lstm_layers = set()
    lstm_pattern = re.compile(r"^lstm\.weight_ih_l(\d+)(?:_reverse)?$")
    for key in state_dict:
        match = lstm_pattern.match(key)
        if match:
            lstm_layers.add(int(match.group(1)))
    if lstm_layers:
        inferred["num_layers"] = max(lstm_layers) + 1

    if "fc.weight" in state_dict:
        inferred["output_size"] = state_dict["fc.weight"].shape[0]

    return inferred


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(payload, nn.Module):
        state_dict = payload.state_dict()
        config_dict = _extract_model_config(payload)
        format_name = "torch_module"
        epoch = None
        metrics = {}
        extra = {}
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        config_dict = payload.get("config", {})
        format_name = "checkpoint"
        epoch = payload.get("epoch")
        metrics = payload.get("metrics", {})
        extra = payload.get("extra", {})
    elif isinstance(payload, dict):
        state_dict = payload
        config_dict = {}
        format_name = "state_dict"
        epoch = None
        metrics = {}
        extra = {}
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(payload)!r}")

    merged_config = {
        **_infer_config_from_state_dict(state_dict),
        **config_dict,
    }
    config = TrainConfig.from_dict(merged_config)

    return {
        "state_dict": state_dict,
        "config": config.to_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "extra": extra,
        "format": format_name,
        "path": str(path),
    }
