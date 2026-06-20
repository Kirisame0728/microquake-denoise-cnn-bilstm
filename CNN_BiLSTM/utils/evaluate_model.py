from __future__ import annotations

from contextlib import nullcontext

import torch


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor.reshape(tensor.size(0), -1)


def calculate_snr_torch(
    clean_signal: torch.Tensor,
    estimate_signal: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    clean_flat = _flatten_batch(clean_signal)
    estimate_flat = _flatten_batch(estimate_signal)

    signal_power = torch.sum(clean_flat ** 2, dim=1).clamp_min(eps)
    noise_power = torch.sum((clean_flat - estimate_flat) ** 2, dim=1).clamp_min(eps)
    return 10.0 * torch.log10(signal_power / noise_power)


def calculate_rmse_torch(clean_signal: torch.Tensor, estimate_signal: torch.Tensor) -> torch.Tensor:
    clean_flat = _flatten_batch(clean_signal)
    estimate_flat = _flatten_batch(estimate_signal)
    return torch.sqrt(torch.mean((clean_flat - estimate_flat) ** 2, dim=1))


def evaluate_model(
    model,
    data_loader,
    device: torch.device,
    use_amp: bool = False,
    return_dict: bool = False,
):
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    snr_before_list = []
    snr_after_list = []
    snr_imp_list = []
    rmse_list = []

    with torch.no_grad():
        for noisy_batch, clean_batch in data_loader:
            noisy_batch = noisy_batch.to(device, non_blocking=device.type == "cuda")
            clean_batch = clean_batch.to(device, non_blocking=device.type == "cuda")

            autocast_context = (
                torch.amp.autocast(device_type="cuda", enabled=True)
                if amp_enabled
                else nullcontext()
            )
            with autocast_context:
                denoised_batch = model(noisy_batch)

            snr_before = calculate_snr_torch(clean_batch, noisy_batch)
            snr_after = calculate_snr_torch(clean_batch, denoised_batch)
            rmse = calculate_rmse_torch(clean_batch, denoised_batch)

            snr_before_list.append(snr_before)
            snr_after_list.append(snr_after)
            snr_imp_list.append(snr_after - snr_before)
            rmse_list.append(rmse)

    metrics = {
        "snr_before": torch.cat(snr_before_list).mean().item(),
        "snr_after": torch.cat(snr_after_list).mean().item(),
        "snr_improvement": torch.cat(snr_imp_list).mean().item(),
        "rmse": torch.cat(rmse_list).mean().item(),
    }

    if return_dict:
        return metrics
    return metrics["snr_before"], metrics["snr_after"], metrics["snr_improvement"]
