from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from CNN_BiLSTM.config import TrainConfig, repo_path
    from CNN_BiLSTM.data_reader import get_full_signal_loader
    from CNN_BiLSTM.model import LSTMCNN
    from CNN_BiLSTM.utils.checkpoint import load_checkpoint
    from CNN_BiLSTM.utils.plot_result import plot_figure
except ImportError:
    from config import TrainConfig, repo_path
    from data_reader import get_full_signal_loader
    from model import LSTMCNN
    from utils.checkpoint import load_checkpoint
    from utils.plot_result import plot_figure


MODEL_OVERRIDE_KEYS = (
    "input_size",
    "hidden_size",
    "num_layers",
    "output_size",
    "dropout_rate",
    "num_cnn_layers",
    "base_channels",
    "max_channels",
    "image_height",
    "image_width",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch prediction for the CNN-BiLSTM denoising model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=4,
        help="Number of signals processed per batch",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=2048,
        help="Waveform length consumed by the model",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        type=str,
        default=repo_path("data", "test"),
        help="Directory containing input signal text files",
    )
    parser.add_argument(
        "--csv-list",
        dest="csv_list",
        type=str,
        default=repo_path("data", "test_list.csv"),
        help="CSV file listing signal file names",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        default=repo_path("pre_trained", "pretrained_denoising_model.pth"),
        help="Path to a checkpoint, state_dict, or serialized torch module",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=repo_path("CNN_BiLSTM", "results"),
        help="Directory used for denoising outputs",
    )
    parser.add_argument(
        "--plot-figure",
        dest="plot_figure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save waveform comparison figures",
    )
    parser.add_argument(
        "--save-signal",
        dest="save_signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save denoised waveforms to text files",
    )
    parser.add_argument(
        "--sampling-rate",
        dest="sampling_rate",
        type=float,
        default=200.0,
        help="Sampling rate used to render the time axis",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device specifier: auto, cpu, cuda, or cuda:0",
    )

    int_overrides = {
        "input_size",
        "hidden_size",
        "num_layers",
        "output_size",
        "num_cnn_layers",
        "base_channels",
        "max_channels",
        "image_height",
        "image_width",
    }
    for key in MODEL_OVERRIDE_KEYS:
        parser.add_argument(
            f"--{key}",
            type=int if key in int_overrides else float,
            default=None,
            help=f"Optional override for checkpoint field {key}",
        )
    return parser


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_inference_config(loaded_config: dict, args: argparse.Namespace) -> TrainConfig:
    overrides = {
        key: getattr(args, key)
        for key in MODEL_OVERRIDE_KEYS
        if getattr(args, key) is not None
    }
    return TrainConfig.from_dict({**loaded_config, **overrides})


def predict(args: argparse.Namespace, device: torch.device) -> Path:
    checkpoint = load_checkpoint(args.model_path, map_location=device)
    config = build_inference_config(checkpoint["config"], args)

    timestamp = time.strftime("pred_%y%m%d-%H%M%S")
    base_out = Path(args.output_dir) / timestamp
    results_dir = base_out / "results"
    figures_dir = base_out / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_figure:
        figures_dir.mkdir(parents=True, exist_ok=True)

    model = LSTMCNN(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    loader = get_full_signal_loader(
        csv_path=args.csv_list,
        dataset_path=args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    logging.info("Loaded %s checkpoint from %s", checkpoint["format"], checkpoint["path"])

    for signals, paths in tqdm(loader, desc="Predicting"):
        for sig_tensor, rel_path in zip(signals, paths):
            signal_length = sig_tensor.size(0)
            denoised_chunks = []

            for start in range(0, signal_length, args.chunk_size):
                raw_chunk = sig_tensor[start:start + args.chunk_size]
                valid_length = raw_chunk.size(0)
                chunk = raw_chunk
                if valid_length < args.chunk_size:
                    pad = args.chunk_size - valid_length
                    chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad))

                model_input = chunk.unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(model_input)
                output = output.squeeze(0).squeeze(-1).cpu().numpy()
                denoised_chunks.append(output[:valid_length])

            denoised = np.concatenate(denoised_chunks, axis=0)
            file_stem = Path(rel_path).stem

            if args.save_signal:
                np.savetxt(results_dir / f"{file_stem}_denoised.txt", denoised, fmt="%.6f")

            if args.plot_figure:
                raw_signal = sig_tensor.squeeze(-1).cpu().numpy()
                time_axis = np.arange(signal_length) / args.sampling_rate
                plot_figure(
                    time=time_axis,
                    noisy_signal=raw_signal,
                    clean_signal=denoised,
                    name=str(figures_dir / f"{file_stem}.png"),
                )

    summary = {
        "checkpoint_format": checkpoint["format"],
        "checkpoint_path": checkpoint["path"],
        "device": str(device),
        "output_dir": str(base_out),
    }
    with (base_out / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    logging.info("Prediction complete. Outputs written to %s", base_out)
    return base_out


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = resolve_device(args.device)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    predict(args, device)


if __name__ == "__main__":
    main()
