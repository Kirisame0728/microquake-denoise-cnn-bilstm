import torch


def calculate_snr_torch(clean_signal: torch.Tensor,
                        noisy_signal: torch.Tensor,
                        eps: float = 1e-10) -> torch.Tensor:

    if clean_signal.dim() > 1:
        clean_flat = clean_signal.view(clean_signal.size(0), -1)
        noisy_flat = noisy_signal.view(clean_signal.size(0), -1)
        signal_power = torch.sum(clean_flat ** 2, dim=1)
        noise_power  = torch.sum((clean_flat - noisy_flat) ** 2, dim=1)
    else:
        clean_flat = clean_signal.view(-1)
        noisy_flat = noisy_signal.view(-1)
        signal_power = torch.sum(clean_flat ** 2)
        noise_power  = torch.sum((clean_flat - noisy_flat) ** 2)

    return 10 * torch.log10(signal_power / (noise_power + eps))


def evaluate_model(model, data_loader, device):
    model.eval()
    snr_before_list = []
    snr_after_list  = []
    snr_imp_list    = []

    with torch.no_grad():
        for noisy_batch, clean_batch in data_loader:
            noisy_batch   = noisy_batch.to(device)
            clean_batch   = clean_batch.to(device)
            denoised_batch = model(noisy_batch)

            B = clean_batch.size(0)
            clean_flat    = clean_batch.view(B, -1)
            noisy_flat    = noisy_batch.view(B, -1)
            denoised_flat = denoised_batch.view(B, -1)

            snr_before = calculate_snr_torch(clean_flat, noisy_flat)      # [B]
            snr_after  = calculate_snr_torch(clean_flat, denoised_flat)   # [B]
            snr_imp    = snr_after - snr_before                           # [B]

            snr_before_list.append(snr_before)
            snr_after_list .append(snr_after)
            snr_imp_list    .append(snr_imp)

    snr_before_all = torch.cat(snr_before_list)
    snr_after_all  = torch.cat(snr_after_list)
    snr_imp_all    = torch.cat(snr_imp_list)

    avg_b = snr_before_all.mean().item()
    avg_a = snr_after_all .mean().item()
    avg_i = snr_imp_all   .mean().item()
    return avg_b, avg_a, avg_i
