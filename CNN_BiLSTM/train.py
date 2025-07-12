import argparse
import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure
from data_reader import *
from tqdm import tqdm
from model import LSTMCNN
from evaluate_model import *


def get_args():
    parser = argparse.ArgumentParser(
        description="Training script for BiLSTM+CNN denoising model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data parameters
    parser.add_argument(
        "--data_path", type=str, default="../data/train/rad_clean_merged.1.npy",
        help="Path to clean data file (.npy)"
    )
    parser.add_argument(
        "--noise_data_path", type=str, default="../data/train/data_noise_30002048_N.npy",
        help="Path to noise data file (.npy)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Proportion of data used for training (rest for validation)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for training"
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

    # Training parameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--scheduler_factor", type=float, default=0.5,
        help="Factor by which the learning rate is reduced"
    )
    parser.add_argument(
        "--scheduler_patience", type=int, default=3,
        help="Number of epochs with no improvement before reducing LR"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="Directory to save training logs and model checkpoints"
    )

    return parser.parse_args()


def set_config(args):

    config = Config()

    config.data_path        = args.data_path
    config.noise_data_path  = args.noise_data_path
    config.train_ratio      = args.train_ratio
    config.batch_size       = args.batch_size

    config.input_size       = args.input_size
    config.hidden_size      = args.hidden_size
    config.num_layers       = args.num_layers
    config.output_size      = args.output_size
    config.dropout_rate     = args.dropout_rate

    config.learning_rate    = args.learning_rate
    config.num_epochs       = args.num_epochs
    config.scheduler_factor = args.scheduler_factor
    config.scheduler_patience = args.scheduler_patience

    config.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    return config


def train(args):
    current_time = time.strftime("%y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, current_time)
    logging.info("Training log: {}".format(log_dir))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    figure_dir = os.path.join(log_dir, 'figures')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    # loss log directory
    loss_log_path = os.path.join(log_dir, 'loss.log')
    loss_fp = open(loss_log_path, 'w')

    writer = SummaryWriter(log_dir=log_dir)

    config = set_config(args)
    with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
        fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMCNN(config).to(device)
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=config.scheduler_factor,
                                                           patience=config.scheduler_patience)
    train_loader, val_loader = get_dataloaders(clean_path='../data/train/rad_clean_merged.1.npy',
                                               noise_path='../data/train/data_noise_30002048_N.npy',
                                               train_ratio=config.train_ratio,
                                               batch_size=config.batch_size,
                                               num_workers=8,
                                               pin_memory=True)

    for epoch in range(config.num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} [Train]", unit="batch")
        model.train()
        total_loss = 0.0

        for noisy, clean in pbar:
            noisy = noisy.to(device, non_blocking=True).view(-1, 1, 32, 64)
            clean = clean.to(device, non_blocking=True).view(-1, 1, 32, 64)

            out = model(noisy).view(-1, 1, 32, 64)
            loss = 1 - ssim_loss(out, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            pbar.set_postfix(train_loss=f"{batch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        avg_train = total_loss / len(train_loader)

        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} [Valid]", unit="batch")
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for noisy, clean in pbar:
                noisy = noisy.to(device, non_blocking=True).view(-1, 1, 32, 64)
                clean = clean.to(device, non_blocking=True).view(-1, 1, 32, 64)

                out = model(noisy).view(-1, 1, 32, 64)
                loss = 1 - ssim_loss(out, clean)

                batch_loss = loss.item()
                total_val += batch_loss
                pbar.set_postfix(val_loss=f"{batch_loss:.4f}")

        avg_val = total_val / len(val_loader)

        train_snr_b, train_snr_a, train_snr_imp = evaluate_model(model, train_loader, device)
        val_snr_b, val_snr_a, val_snr_imp = evaluate_model(model, val_loader, device)

        line = (
            f"Epoch {epoch + 1}  "
            f"Train_Loss {avg_train:.6f}  "
            f"Val_Loss {avg_val:.6f}  "
            f"Train_SNR_Before {train_snr_b:.2f}  "
            f"Train_SNR_After {train_snr_a:.2f}  "
            f"Train_SNR_Imp {train_snr_imp:.2f}  "
            f"Val_SNR_Before {val_snr_b:.2f}  "
            f"Val_SNR_After {val_snr_a:.2f}  "
            f"Val_SNR_Imp {val_snr_imp:.2f}"
        )

        tqdm.write(line)
        loss_fp.write(line + "\n")

        # Write scalars to TensorBoard
        writer.add_scalars('Loss', {'Train': avg_train,
                                   'Validation': avg_val}, epoch+1)
        writer.add_scalars('SNR', {'Train': train_snr_a,
                                   'Validation': val_snr_a,
                                   'Train_SNR_Before': train_snr_b,
                                   'Validation_SNR_Before': val_snr_b}, epoch+1)
        writer.add_scalars('SNR_improvement', {'Train': train_snr_imp,
                                           'Validation': val_snr_imp}, epoch+1)

        scheduler.step(avg_val)

    loss_fp.close()
    writer.close()

    model_dir = os.path.join(log_dir, "whole_model.pth")
    torch.save(model, model_dir)


if __name__ == '__main__':
    train(get_args())