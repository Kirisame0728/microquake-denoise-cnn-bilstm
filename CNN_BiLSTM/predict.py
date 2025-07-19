#!/usr/bin/env python
import argparse
import logging
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from model import LSTMCNN
from data_reader import get_full_signal_loader

from utils.plot_result import plot_figure

def parse_args():
    parser = argparse.ArgumentParser(description="Batch prediction for seismic denoising model")
    parser.add_argument("--batch_size",    type=int,   default=4,     help="Number of signals per batch")
    parser.add_argument("--chunk_size",    type=int,   default=2048,  help="Length of each chunk for the model input")
    parser.add_argument("--data_dir",      type=str,   default="../data/test/",   help="Directory containing signal files")
    parser.add_argument("--csv_list",      type=str,   default="../data/test_list.csv", help="CSV file listing signal file paths")
    parser.add_argument("--model_path",    type=str,   default="../pre_trained/pretrained_denoising_model.pth", help="Path to the trained .pth model file")
    parser.add_argument("--output_dir",    type=str,   default="results/",    help="Directory to save predictions")
    parser.add_argument("--plot_figure",   action="store_true", default=True, help="Whether to save comparison plots")
    parser.add_argument("--save_signal",   action="store_true", default=True, help="Whether to save denoised signals")
    parser.add_argument("--sampling_rate", type=float, default=200.0,        help="Sampling rate (Hz)")

    # Model parameters
    parser.add_argument("--input_size",   type=int,   default=1,   help="Dimensionality of input features")
    parser.add_argument("--hidden_size",  type=int,   default=64,  help="Hidden dimension size of LSTM layers")
    parser.add_argument("--num_layers",   type=int,   default=2,   help="Number of LSTM layers")
    parser.add_argument("--output_size",  type=int,   default=1,   help="Dimensionality of model outputs")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout probability")

    return parser.parse_args()


def predict(args, device):
    args.device = device

    now = time.strftime("%y%m%d-%H%M%S")
    base_out = os.path.join(args.output_dir, f"pred_{now}")
    os.makedirs(base_out, exist_ok=True)
    res_dir = os.path.join(base_out, "results")
    os.makedirs(res_dir, exist_ok=True)
    fig_dir = os.path.join(base_out, "figures") if args.plot_figure else None
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)

    model = LSTMCNN(args)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logging.info(f"Loaded model from {args.model_path}")

    loader = get_full_signal_loader(args.csv_list,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    dataset_path="../data/test/")

    for signals, paths in tqdm(loader, desc="Predicting"):
        for sig_tensor, path in zip(signals, paths):
            length = sig_tensor.size(0)
            denoised_chunks = []

            # chunk & infer
            for i in range(0, length, args.chunk_size):
                chunk = sig_tensor[i:i + args.chunk_size]     # (≤chunk_size,1)
                if chunk.size(0) < args.chunk_size:
                    pad = args.chunk_size - chunk.size(0)
                    chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad))
                inp = chunk.unsqueeze(0).to(device)           # (1,chunk_size,1)
                with torch.no_grad():
                    out = model(inp)                          # (1,chunk_size,1)
                out = out.squeeze(0).squeeze(-1).cpu().numpy()  # (chunk_size,)
                valid_len = min(chunk.size(0), length - i)
                denoised_chunks.append(out[:valid_len])

            # reconstruct
            denoised = np.concatenate(denoised_chunks, axis=0)  # (length,)

            fname = os.path.splitext(os.path.basename(path))[0]
            if args.save_signal:
                den_path = os.path.join(res_dir, fname + "_denoised.txt")
                np.savetxt(den_path, denoised, fmt="%.6f")

            if args.plot_figure:
                raw = sig_tensor.squeeze(-1).cpu().numpy()
                t = np.arange(length) / args.sampling_rate
                plot_path = os.path.join(fig_dir, fname + ".png")
                plot_figure(t, raw, denoised, plot_path)

    logging.info(f"Done. All outputs under {base_out}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    predict(args, device)


if __name__ == "__main__":
    main()
