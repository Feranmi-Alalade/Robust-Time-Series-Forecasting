import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--log",
    default="grid_search_results/cyclenettrend_loss_curves.txt"
)
parser.add_argument(
    "--output",
    default="grid_search_results/cyclenettrend_loss_plots"
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Also plot every hyperparameter configuration"
)
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Patterns matching the actual CycleNetTrend logs
# ============================================================

header_pattern = re.compile(
    r"start training\s*:\s*(.*)"
)

epoch_pattern = re.compile(
    r"Epoch:\s*(\d+),\s*Steps:\s*\d+\s*\|"
    r"\s*Train Loss:\s*([0-9.eE+-]+)"
    r"\s+Vali Loss:\s*([0-9.eE+-]+)"
)

config_pattern = re.compile(
    r"^(.*?)_CycleNetTrend_(\d+)_(\d+)"
    r"_tk(\d+)_tl(\d+)_lr([0-9.eE+-]+?)_"
)

# ============================================================
# 2. Read and separate individual training runs
# ============================================================

runs = []
current = None

with open(args.log, "r", encoding="utf-8", errors="replace") as file:
    for line in file:

        header = header_pattern.search(line)

        if header:
            if current and current["epochs"]:
                runs.append(current)

            setting = header.group(1).strip().rstrip(">")

            match = config_pattern.search(setting)

            if not match:
                print(f"Unrecognized configuration: {setting}")
                current = None
                continue

            dataset, lookback, horizon, kernel, layers, lr = match.groups()

            current = {
                "setting": setting,
                "dataset": dataset,
                "lookback": int(lookback),
                "horizon": int(horizon),
                "kernel": int(kernel),
                "layers": int(layers),
                "lr": float(lr),
                "epochs": [],
                "train_loss": [],
                "val_loss": []
            }

            continue

        match = epoch_pattern.search(line)

        if match and current is not None:

            epoch, train, val = match.groups()

            current["epochs"].append(int(epoch))
            current["train_loss"].append(float(train))
            current["val_loss"].append(float(val))

if current and current["epochs"]:
    runs.append(current)

print(f"\nExtracted {len(runs)} training runs.")

if not runs:
    raise RuntimeError("No CycleNetTrend training runs were found.")

# ============================================================
# 3. Calculate summary statistics
# ============================================================

for run in runs:

    best_index = min(
        range(len(run["val_loss"])),
        key=lambda i: run["val_loss"][i]
    )

    run["best_epoch"] = run["epochs"][best_index]
    run["best_val"] = run["val_loss"][best_index]
    run["initial_train"] = run["train_loss"][0]
    run["final_train"] = run["train_loss"][-1]

    run["train_reduction_pct"] = (
        (run["initial_train"] - run["final_train"])
        / run["initial_train"] * 100
    )

# ============================================================
# 4. Save detailed summary CSV
# ============================================================

fields = [
    "dataset",
    "horizon",
    "lookback",
    "kernel",
    "layers",
    "lr",
    "best_epoch",
    "best_val",
    "initial_train",
    "final_train",
    "train_reduction_pct",
    "setting"
]

with open(output_dir / "all_runs_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for run in runs:
        writer.writerow({key: run[key] for key in fields})

# ============================================================
# 5. Function to draw loss curves
# ============================================================

def plot_run(run, filename):

    epochs = run["epochs"]
    train = run["train_loss"]
    val = run["val_loss"]

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        epochs,
        train,
        marker="o",
        markersize=3,
        label="Training loss"
    )

    plt.plot(
        epochs,
        val,
        marker="s",
        markersize=3,
        label="Validation loss"
    )

    plt.axvline(
        run["best_epoch"],
        linestyle="--",
        alpha=0.7,
        label=f"Best validation epoch: {run['best_epoch']}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    plt.title(
        f"CycleNetTrend — {run['dataset']} — H={run['horizon']}\n"
        f"Kernel={run['kernel']}, Layers={run['layers']}, "
        f"LR={run['lr']:g}"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / filename, dpi=300)
    plt.close()

# ============================================================
# 6. Select best configuration using validation loss ONLY
# ============================================================

groups = defaultdict(list)

for run in runs:
    key = (run["dataset"], run["horizon"])
    groups[key].append(run)

best_runs = []

for (dataset, horizon), candidates in sorted(groups.items()):

    best = min(candidates, key=lambda r: r["best_val"])
    best_runs.append(best)

    filename = f"BEST_{dataset}_H{horizon}.png"

    plot_run(best, filename)

    print(
        f"\nDataset: {dataset} | H={horizon}"
        f"\n  Kernel: {best['kernel']}"
        f"\n  Layers: {best['layers']}"
        f"\n  LR: {best['lr']}"
        f"\n  Best validation epoch: {best['best_epoch']}"
        f"\n  Best validation loss: {best['best_val']:.6f}"
        f"\n  Initial training loss: {best['initial_train']:.6f}"
        f"\n  Final training loss: {best['final_train']:.6f}"
        f"\n  Training loss reduction: "
        f"{best['train_reduction_pct']:.2f}%"
    )

with open(output_dir / "best_configurations.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for run in best_runs:
        writer.writerow({key: run[key] for key in fields})

# ============================================================
# 7. Optional: plot every configuration
# ============================================================

if args.all:

    all_dir = output_dir / "all_configurations"
    all_dir.mkdir(exist_ok=True)

    for i, run in enumerate(runs, start=1):

        filename = (
            f"all_configurations/run{i:03d}_"
            f"{run['dataset']}_H{run['horizon']}_"
            f"tk{run['kernel']}_tl{run['layers']}_"
            f"lr{run['lr']:g}.png"
        )

        plot_run(run, filename)

print(f"\nResults saved to: {output_dir}")