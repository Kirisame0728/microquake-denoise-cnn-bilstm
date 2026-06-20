from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from tqdm import tqdm

try:
    from CNN_BiLSTM.config import TrainConfig, build_train_arg_parser, config_from_args
    from CNN_BiLSTM.data_reader import get_dataloaders
    from CNN_BiLSTM.model import LSTMCNN
    from CNN_BiLSTM.utils.checkpoint import save_checkpoint
    from CNN_BiLSTM.utils.evaluate_model import evaluate_model
except ImportError:
    from config import TrainConfig, build_train_arg_parser, config_from_args
    from data_reader import get_dataloaders
    from model import LSTMCNN
    from utils.checkpoint import save_checkpoint
    from utils.evaluate_model import evaluate_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def create_run_dir(base_dir: str | Path, run_name: str | None = None) -> Path:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = time.strftime("train_%y%m%d-%H%M%S")
    run_dir = base_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def reshape_for_ssim(batch: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    sample_length = batch[0].numel()
    if sample_length != config.sample_length:
        raise ValueError(
            "Waveform length does not match the SSIM reshape target: "
            f"expected {config.sample_length}, got {sample_length}"
        )
    return batch.reshape(batch.size(0), 1, config.image_height, config.image_width)


def run_epoch(
    model: LSTMCNN,
    data_loader,
    optimizer,
    device: torch.device,
    config: TrainConfig,
    amp_enabled: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch_label: str,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(data_loader, desc=epoch_label, unit="batch")
    for noisy, clean in progress:
        noisy = noisy.to(device, non_blocking=device.type == "cuda")
        clean = clean.to(device, non_blocking=device.type == "cuda")

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_training else torch.no_grad()
        autocast_context = (
            torch.amp.autocast(device_type="cuda", enabled=True)
            if amp_enabled
            else nullcontext()
        )

        with grad_context:
            with autocast_context:
                out = model(noisy)
                loss = 1.0 - structural_similarity_index_measure(
                    reshape_for_ssim(out, config),
                    reshape_for_ssim(clean, config),
                    data_range=2.0,
                )

        if is_training and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_loss = loss.item()
        total_loss += batch_loss
        postfix = {"loss": f"{batch_loss:.4f}"}
        if is_training:
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.1e}"
        progress.set_postfix(postfix)

    return total_loss / len(data_loader)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def create_summary_writer(log_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:  # pragma: no cover - depends on local tensorboard install
        logging.warning("TensorBoard logging disabled: %s", exc)
        return None


def train_model(
    config: TrainConfig,
    run_dir: str | Path | None = None,
    tensorboard: bool = True,
) -> tuple[dict, Path]:
    config = config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    amp_enabled = config.amp and device.type == "cuda"

    run_dir_path = Path(run_dir) if run_dir is not None else create_run_dir(config.log_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    logging.info("Using device: %s", device)
    logging.info("Run directory: %s", run_dir_path)
    logging.info("AMP enabled: %s", amp_enabled)

    write_json(run_dir_path / "config.json", config.to_dict())

    writer = create_summary_writer(run_dir_path, tensorboard)
    history: list[dict] = []
    best_state_dict = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_start = time.perf_counter()

    train_loader, val_loader, test_loader = get_dataloaders(
        clean_path=config.data_path,
        noise_path=config.noise_data_path,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        noise_scale=config.noise_scale,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
    )

    model = LSTMCNN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(config.num_epochs):
        epoch_index = epoch + 1
        epoch_start = time.perf_counter()

        avg_train = run_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            amp_enabled=amp_enabled,
            scaler=scaler,
            epoch_label=f"Epoch {epoch_index}/{config.num_epochs} [Train]",
        )
        avg_val = run_epoch(
            model=model,
            data_loader=val_loader,
            optimizer=None,
            device=device,
            config=config,
            amp_enabled=amp_enabled,
            scaler=None,
            epoch_label=f"Epoch {epoch_index}/{config.num_epochs} [Valid]",
        )

        train_metrics = evaluate_model(
            model=model,
            data_loader=train_loader,
            device=device,
            use_amp=amp_enabled,
            return_dict=True,
        )
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            use_amp=amp_enabled,
            return_dict=True,
        )

        scheduler.step(avg_val)
        epoch_seconds = time.perf_counter() - epoch_start
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else None
        )

        record = {
            "epoch": epoch_index,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_mb": peak_memory_mb,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(record)

        logging.info(
            "Epoch %d | train_loss %.6f | val_loss %.6f | "
            "train_snr %.2f dB | val_snr %.2f dB | rmse %.5f",
            epoch_index,
            avg_train,
            avg_val,
            train_metrics["snr_after"],
            val_metrics["snr_after"],
            val_metrics["rmse"],
        )

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train, epoch_index)
            writer.add_scalar("Loss/val", avg_val, epoch_index)
            writer.add_scalar("SNR/train_before", train_metrics["snr_before"], epoch_index)
            writer.add_scalar("SNR/train_after", train_metrics["snr_after"], epoch_index)
            writer.add_scalar("SNR/train_improvement", train_metrics["snr_improvement"], epoch_index)
            writer.add_scalar("SNR/val_before", val_metrics["snr_before"], epoch_index)
            writer.add_scalar("SNR/val_after", val_metrics["snr_after"], epoch_index)
            writer.add_scalar("SNR/val_improvement", val_metrics["snr_improvement"], epoch_index)
            writer.add_scalar("RMSE/val", val_metrics["rmse"], epoch_index)
            writer.add_scalar("Time/epoch_seconds", epoch_seconds, epoch_index)
            if peak_memory_mb is not None:
                writer.add_scalar("Memory/peak_gpu_memory_mb", peak_memory_mb, epoch_index)

        improved = avg_val < (best_val_loss - config.early_stopping_min_delta)
        if improved:
            best_val_loss = avg_val
            best_epoch = epoch_index
            best_state_dict = deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir_path / "best_checkpoint.pth",
                model=model,
                config=config,
                epoch=epoch_index,
                metrics=record,
                extra={"amp_enabled": amp_enabled},
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            logging.info(
                "Early stopping triggered at epoch %d after %d stale epochs.",
                epoch_index,
                epochs_without_improvement,
            )
            break

    if best_state_dict is None:
        best_state_dict = deepcopy(model.state_dict())
        best_epoch = len(history)
        best_val_loss = history[-1]["val_loss"]

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        use_amp=amp_enabled,
        return_dict=True,
    )

    total_seconds = time.perf_counter() - train_start
    save_checkpoint(
        run_dir_path / "last_checkpoint.pth",
        model=model,
        config=config,
        epoch=len(history),
        metrics=history[-1],
        extra={"amp_enabled": amp_enabled},
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history),
        "train_time_minutes": total_seconds / 60.0,
        "amp_enabled": amp_enabled,
        "device": str(device),
        "peak_gpu_memory_mb": max(
            (entry["peak_gpu_memory_mb"] or 0.0) for entry in history
        ),
        "test_metrics": test_metrics,
        "best_checkpoint": str(run_dir_path / "best_checkpoint.pth"),
        "last_checkpoint": str(run_dir_path / "last_checkpoint.pth"),
    }

    write_json(run_dir_path / "history.json", {"epochs": history})
    write_json(run_dir_path / "summary.json", summary)

    if writer is not None:
        writer.add_hparams(
            {
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "dropout_rate": config.dropout_rate,
                "num_cnn_layers": config.num_cnn_layers,
                "learning_rate": config.learning_rate,
                "amp": int(amp_enabled),
            },
            {
                "hparam/best_val_loss": best_val_loss,
                "hparam/test_snr_after": test_metrics["snr_after"],
                "hparam/test_rmse": test_metrics["rmse"],
            },
        )
        writer.close()

    logging.info(
        "Training complete in %.2f minutes. Best epoch: %d. Test SNR after denoising: %.2f dB",
        summary["train_time_minutes"],
        best_epoch,
        test_metrics["snr_after"],
    )
    return summary, run_dir_path


def main() -> None:
    parser = build_train_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    train_model(config)


if __name__ == "__main__":
    main()
