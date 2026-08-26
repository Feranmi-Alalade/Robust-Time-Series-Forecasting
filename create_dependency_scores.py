import argparse
import os

import numpy as np
import torch

from data_provider.data_factory import data_provider
from models import CycleNet


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a D x D dependency-score matrix from CycleNet training residuals."
    )

    # Dataset
    parser.add_argument("--root_path", type=str, default="./dataset/")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")

    # Forecasting setup
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--enc_in", type=int, required=True)

    # CycleNet
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--model_type", type=str, default="mlp")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--use_revin", type=int, default=1)

    # Loader
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=10)

    # Pretrained CycleNet checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained CycleNet checkpoint.pth"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npy path, e.g. ./dependency_scores/weather_96.npy"
    )

    return parser


def load_cyclenet(args, device):
    model = CycleNet.Model(args).float().to(device)

    state = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(state)
    model.eval()

    return model


def collect_training_residuals(model, train_loader, args, device):
    residuals = []

    with torch.no_grad():
        for batch_x, _, _, _, batch_cycle in train_loader:
            x = batch_x.float().to(device)
            cycle_index = batch_cycle.int().to(device)

            # Match CycleNet's preprocessing exactly.
            if args.use_revin:
                seq_mean = torch.mean(
                    x,
                    dim=1,
                    keepdim=True
                )

                seq_var = (
                    torch.var(
                        x,
                        dim=1,
                        keepdim=True
                    )
                    + 1e-5
                )

                x = (
                    x - seq_mean
                ) / torch.sqrt(seq_var)

            cycle = model.cycleQueue(
                cycle_index,
                args.seq_len
            )

            residual = x - cycle

            # B x L x D -> (B*L) x D
            residuals.append(
                residual.reshape(
                    -1,
                    args.enc_in
                ).cpu()
            )

    return torch.cat(
        residuals,
        dim=0
    ).numpy()


def compute_dependency_scores(residuals):
    """
    Absolute Pearson correlation across residual channels.

    Output:
        D x D matrix

    score[i, j] represents the strength of dependency between
    target channel i and source channel j.

    The diagonal is set to zero because PairMix selects only
    OTHER channels.
    """

    scores = np.corrcoef(
        residuals,
        rowvar=False
    )

    scores = np.abs(scores)

    # Handle constant channels / numerical problems.
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    np.fill_diagonal(
        scores,
        0.0
    )

    return scores.astype(
        np.float32
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Use the repository's normal training split.
    _, train_loader = data_provider(
        args,
        flag="train"
    )

    model = load_cyclenet(
        args,
        device
    )

    residuals = collect_training_residuals(
        model,
        train_loader,
        args,
        device
    )

    print(
        f"Residual matrix: {residuals.shape}"
    )

    scores = compute_dependency_scores(
        residuals
    )

    print(
        f"Dependency matrix: {scores.shape}"
    )

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True
    )

    np.save(
        args.output,
        scores
    )

    print(
        f"Saved dependency scores to: {args.output}"
    )


if __name__ == "__main__":
    main()