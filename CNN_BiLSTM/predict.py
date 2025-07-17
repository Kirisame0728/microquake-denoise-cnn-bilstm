import argparse
import logging
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from model import LSTMCNN


def get_args():
    parser = argparse.ArgumentParser(
        description="Prediction script for BiLSTM+CNN denoising model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Paths with defaults
    parser.add_argument(
        "--model_path", type=str, default="..pre_trained/pretrained_denoising_model.pth",
        help="Path to the trained model .pth file"
    )
    parser.add_argument(
        "--input_path", type=str, default="../data/noisy_input.txt",
        help="Path to the input noisy signal (.txt or .npy)"
    )
    parser.add_argument(
        "--output_path", type=str, default="../results/denoised_output.txt",
        help="Path to save the denoised signal (.txt)"
    )
    # Prediction parameters
    parser.add_argument(
        "--chunk_size", type=int, default=2048,
        help="Length of each chunk for block-wise denoising"
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Whether to normalize input signal amplitude to [-1,1]"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (cpu or cuda)"
    )

    # Model parameters
    parser.add_argument(
        "--input_size", type=int, default=1,
        help="Dimensionality of input features"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=64,
        help="Hidden dimension size of LSTM layers"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2,
        help="Number of LSTM layers"
    )
    parser.add_argument(
        "--output_size", type=int, default=1,
        help="Dimensionality of model outputs"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.1,
        help="Dropout probability"
    )

    
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_signal(path):
    ext = os.path.splitext(path)[1]
    if ext in ['.txt', '.dat']:
        data = np.loadtxt(path)
    elif ext in ['.npy']:
        data = np.load(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return data


def normalize_amplitude(signal):
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val != 0 else signal


def denoise_signal(model, signal, chunk_size, device):
    model.eval()
    length = signal.shape[0]
    segments = []

    with torch.no_grad():
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            chunk = signal[start:end]
            if end - start < chunk_size:
                pad = chunk_size - (end - start)
                chunk = F.pad(
                    torch.tensor(chunk, dtype=torch.float32),
                    (0, pad), mode='constant', value=0
                )
            else:
                chunk = torch.tensor(chunk, dtype=torch.float32)

            tensor = chunk.unsqueeze(0).unsqueeze(-1).to(device)
            output = model(tensor)
            out_chunk = output.squeeze().cpu().numpy()
            segments.append(out_chunk[: end - start])

    return np.concatenate(segments, axis=0)


def main():
    args = get_args()
    setup_logging()
    logging.info("Starting prediction with model: %s", args.model_path)

    # Load model
    device = torch.device(args.device)
    model = LSTMCNN()
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Load data
    logging.info("Loading input signal from %s", args.input_path)
    signal = load_signal(args.input_path)
    if args.normalize:
        signal = normalize_amplitude(signal)

    # Denoise
    logging.info("Denoising signal in chunks of size %d", args.chunk_size)
    denoised = denoise_signal(model, signal, args.chunk_size, device)

    # Save
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    np.savetxt(args.output_path, denoised, fmt="%.6f")
    logging.info("Denoised signal saved to %s", args.output_path)


if __name__ == '__main__':
    main()