from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

try:
    from CNN_BiLSTM.config import TrainConfig, build_train_arg_parser, config_from_args, repo_path
    from CNN_BiLSTM.train import train_model
except ImportError:
    from config import TrainConfig, build_train_arg_parser, config_from_args, repo_path
    from train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = build_train_arg_parser(
        description="Bayesian optimization for CNN-BiLSTM hyperparameters"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=4,
        help="Total number of Bayesian optimization trials",
    )
    parser.add_argument(
        "--random-starts",
        dest="random_starts",
        type=int,
        default=2,
        help="Number of random warm-up trials before fitting the Gaussian process",
    )
    parser.add_argument(
        "--candidate-pool",
        dest="candidate_pool",
        type=int,
        default=256,
        help="Number of random candidates scored by Expected Improvement per BO step",
    )
    parser.add_argument(
        "--search-log-dir",
        dest="search_log_dir",
        type=str,
        default=repo_path("CNN_BiLSTM", "logs", "bayes_search"),
        help="Directory used for search logs and trial artifacts",
    )
    parser.add_argument(
        "--train-best-model",
        dest="train_best_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retrain the best hyperparameter setting after the search finishes",
    )

    parser.add_argument("--cnn-layers-min", dest="cnn_layers_min", type=int, default=2)
    parser.add_argument("--cnn-layers-max", dest="cnn_layers_max", type=int, default=10)
    parser.add_argument("--hidden-size-min", dest="hidden_size_min", type=int, default=16)
    parser.add_argument("--hidden-size-max", dest="hidden_size_max", type=int, default=512)
    parser.add_argument("--dropout-min", dest="dropout_min", type=float, default=0.0)
    parser.add_argument("--dropout-max", dest="dropout_max", type=float, default=0.5)
    parser.add_argument("--lr-min", dest="lr_min", type=float, default=1e-4)
    parser.add_argument("--lr-max", dest="lr_max", type=float, default=1e-2)
    return parser


def create_search_dir(base_dir: str | Path) -> Path:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    run_dir = base_path / time.strftime("search_%y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sample_parameters(args: argparse.Namespace, rng: np.random.Generator) -> dict[str, float | int]:
    hidden_low = math.ceil(args.hidden_size_min / 16)
    hidden_high = math.floor(args.hidden_size_max / 16)
    hidden_units = int(rng.integers(hidden_low, hidden_high + 1) * 16)

    return {
        "num_cnn_layers": int(rng.integers(args.cnn_layers_min, args.cnn_layers_max + 1)),
        "hidden_size": hidden_units,
        "dropout_rate": float(rng.uniform(args.dropout_min, args.dropout_max)),
        "learning_rate": float(
            10 ** rng.uniform(np.log10(args.lr_min), np.log10(args.lr_max))
        ),
    }


def encode_parameters(args: argparse.Namespace, params: dict[str, float | int]) -> np.ndarray:
    return np.array(
        [
            (params["num_cnn_layers"] - args.cnn_layers_min)
            / max(args.cnn_layers_max - args.cnn_layers_min, 1),
            (params["hidden_size"] - args.hidden_size_min)
            / max(args.hidden_size_max - args.hidden_size_min, 1),
            (params["dropout_rate"] - args.dropout_min)
            / max(args.dropout_max - args.dropout_min, 1e-12),
            (np.log10(params["learning_rate"]) - np.log10(args.lr_min))
            / max(np.log10(args.lr_max) - np.log10(args.lr_min), 1e-12),
        ],
        dtype=float,
    )


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_value: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    improvement = best_value - mu
    z_score = improvement / sigma
    ei = improvement * norm.cdf(z_score) + sigma * norm.pdf(z_score)
    ei[sigma <= 1e-12] = 0.0
    return ei


def parameter_signature(params: dict[str, float | int]) -> tuple:
    return (
        int(params["num_cnn_layers"]),
        int(params["hidden_size"]),
        round(float(params["dropout_rate"]), 6),
        round(float(params["learning_rate"]), 8),
    )


def propose_parameters(
    args: argparse.Namespace,
    rng: np.random.Generator,
    tried_signatures: set[tuple],
    encoded_points: list[np.ndarray],
    objective_values: list[float],
) -> dict[str, float | int]:
    if len(encoded_points) < max(2, args.random_starts):
        while True:
            params = sample_parameters(args, rng)
            if parameter_signature(params) not in tried_signatures:
                return params

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(4), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=args.seed,
        n_restarts_optimizer=3,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(np.vstack(encoded_points), np.array(objective_values, dtype=float))

    candidates: list[dict[str, float | int]] = []
    candidate_vectors: list[np.ndarray] = []
    while len(candidates) < args.candidate_pool:
        params = sample_parameters(args, rng)
        signature = parameter_signature(params)
        if signature in tried_signatures:
            continue
        candidates.append(params)
        candidate_vectors.append(encode_parameters(args, params))

    candidate_matrix = np.vstack(candidate_vectors)
    mu, sigma = gp.predict(candidate_matrix, return_std=True)
    best_value = min(objective_values)
    ei = expected_improvement(mu, sigma, best_value)
    return candidates[int(np.argmax(ei))]


def write_trial_table(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "trial",
        "num_cnn_layers",
        "hidden_size",
        "dropout_rate",
        "learning_rate",
        "best_val_loss",
        "train_time_minutes",
        "trial_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    rng = np.random.default_rng(args.seed)
    search_dir = create_search_dir(args.search_log_dir)
    tried_signatures: set[tuple] = set()
    encoded_points: list[np.ndarray] = []
    objective_values: list[float] = []
    trial_rows: list[dict] = []

    best_params: dict[str, float | int] | None = None
    best_summary: dict | None = None

    for trial_idx in range(1, args.trials + 1):
        params = propose_parameters(
            args=args,
            rng=rng,
            tried_signatures=tried_signatures,
            encoded_points=encoded_points,
            objective_values=objective_values,
        )
        tried_signatures.add(parameter_signature(params))

        trial_config = replace(
            config,
            num_cnn_layers=int(params["num_cnn_layers"]),
            hidden_size=int(params["hidden_size"]),
            dropout_rate=float(params["dropout_rate"]),
            learning_rate=float(params["learning_rate"]),
        )
        trial_dir = search_dir / f"trial_{trial_idx:02d}"
        logging.info("Trial %d/%d with params: %s", trial_idx, args.trials, params)

        summary, _ = train_model(trial_config, run_dir=trial_dir, tensorboard=False)
        best_val_loss = float(summary["best_val_loss"])

        encoded_points.append(encode_parameters(args, params))
        objective_values.append(best_val_loss)

        row = {
            "trial": trial_idx,
            "num_cnn_layers": params["num_cnn_layers"],
            "hidden_size": params["hidden_size"],
            "dropout_rate": params["dropout_rate"],
            "learning_rate": params["learning_rate"],
            "best_val_loss": best_val_loss,
            "train_time_minutes": summary["train_time_minutes"],
            "trial_dir": str(trial_dir),
        }
        trial_rows.append(row)

        if best_summary is None or best_val_loss < float(best_summary["best_val_loss"]):
            best_params = params
            best_summary = summary

    write_trial_table(search_dir / "trials.csv", trial_rows)
    with (search_dir / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2)

    search_summary = {
        "best_params": best_params,
        "best_summary": best_summary,
        "trials_csv": str(search_dir / "trials.csv"),
    }

    if args.train_best_model and best_params is not None:
        final_config = replace(
            config,
            num_cnn_layers=int(best_params["num_cnn_layers"]),
            hidden_size=int(best_params["hidden_size"]),
            dropout_rate=float(best_params["dropout_rate"]),
            learning_rate=float(best_params["learning_rate"]),
        )
        final_summary, final_dir = train_model(
            final_config,
            run_dir=search_dir / "best_model",
            tensorboard=True,
        )
        search_summary["final_training"] = {
            "summary": final_summary,
            "run_dir": str(final_dir),
        }

    with (search_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(search_summary, file, indent=2)

    logging.info("Bayesian search complete. Results written to %s", search_dir)


if __name__ == "__main__":
    main()
