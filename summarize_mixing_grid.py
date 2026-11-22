
#!/usr/bin/env python3
"""Select CycleNet mixing configurations by lowest *epoch* validation MSE.

Run from the CycleNet repository root:
    python -u summarize_mixing_grid.py Solar
    python -u summarize_mixing_grid.py Weather
    python -u summarize_mixing_grid.py ETTm1

The dataset grid-search scripts call this automatically after all trials complete.
"""

import csv
import math
import re
import sys
from pathlib import Path

DATASETS = {"Solar": "solar", "Weather": "weather", "ETTm1": "ettm1"}
HORIZONS = (96, 192, 336, 720)
EXPECTED_PER_HORIZON = {"CycleNetMix": 32, "CycleNetPairMix": 48, "CycleNetMixLowRank": 96}

FILE_RE = re.compile(
    r"^(Solar|Weather|ETTm1)_(CycleNetMixLowRank|CycleNetPairMix|CycleNetMix)_"
    r"L(96|192|336|720)_H(96|192|336|720)_(.+)\.log$"
)
# Both versions of the training code are supported.
EPOCH_VAL_RE = re.compile(
    r"^\s*Epoch:\s*(\d+),\s*Steps:[^\r\n]*?"
    r"(?:Val MSE|Vali Loss):\s*([0-9.eE+-]+)",
    re.MULTILINE,
)
# Accept both one-line "mse:..., mae:..." and two-line "MSE:"/"MAE:".
MSE_RE = re.compile(r"^[ \t]*MSE:\s*([0-9.eE+-]+)", re.IGNORECASE | re.MULTILINE)
MAE_RE = re.compile(r"(?:^[ \t]*|,[ \t]*)MAE:\s*([0-9.eE+-]+)", re.IGNORECASE | re.MULTILINE)

COLUMNS = (
    "dataset", "model", "horizon", "lookback", "hyperparameters",
    "best_epoch", "validation_mse", "test_mse", "test_mae", "run_id",
)


def parse_log(log: Path):
    content = log.read_text(encoding="utf-8", errors="replace")
    epochs = [
        (int(epoch), float(value))
        for epoch, value in EPOCH_VAL_RE.findall(content)
        if math.isfinite(float(value))
    ]
    mses = [float(x) for x in MSE_RE.findall(content) if math.isfinite(float(x))]
    maes = [float(x) for x in MAE_RE.findall(content) if math.isfinite(float(x))]
    if not epochs or not mses or not maes:
        raise ValueError("missing or nonfinite epoch validation MSE / final test MSE / final test MAE")
    best_epoch, best_val = min(epochs, key=lambda item: item[1])
    return best_epoch, best_val, mses[-1], maes[-1]


def write_csv(path: Path, records):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in DATASETS:
        sys.exit("Usage: python summarize_mixing_grid.py Solar|Weather|ETTm1")

    dataset = sys.argv[1]
    directory = Path(f"{DATASETS[dataset]}_grid_results_full")
    if not directory.is_dir():
        sys.exit(f"ERROR: missing results directory: {directory}")

    all_csv = directory / "all_completed_trials.csv"
    best_csv = directory / "best_by_validation_mse.csv"
    # Never leave a stale 'complete' summary after an incomplete rerun.
    best_csv.unlink(missing_ok=True)

    records, errors = [], []
    for log in sorted(directory.glob("*.log")):
        match = FILE_RE.fullmatch(log.name)
        if match is None or match.group(1) != dataset:
            errors.append(f"Unexpected log filename: {log.name}")
            continue
        if not log.with_suffix(".done").is_file():
            errors.append(f"No .done marker: {log.name}")
            continue
        ds, model, lookback, horizon, params = match.groups()
        try:
            epoch, val_mse, test_mse, test_mae = parse_log(log)
        except ValueError as err:
            errors.append(f"{log.name}: {err}")
            continue
        records.append({
            "dataset": ds, "model": model, "horizon": int(horizon),
            "lookback": int(lookback), "hyperparameters": params,
            "best_epoch": epoch, "validation_mse": val_mse,
            "test_mse": test_mse, "test_mae": test_mae,
            "run_id": log.stem,
        })

    records.sort(key=lambda r: (r["model"], r["horizon"], r["lookback"], r["run_id"]))
    write_csv(all_csv, records)

    selected = []
    for model, expected in EXPECTED_PER_HORIZON.items():
        for horizon in HORIZONS:
            group = [r for r in records if r["model"] == model and r["horizon"] == horizon]
            if len(group) != expected:
                errors.append(f"{model} H={horizon}: expected {expected} trials, found {len(group)}")
                continue
            # No test metric is involved in selection.
            selected.append(min(group, key=lambda r: (r["validation_mse"], r["run_id"])))

    expected_total = sum(EXPECTED_PER_HORIZON.values()) * len(HORIZONS)
    print(f"{dataset}: {len(records)}/{expected_total} valid completed experiments", flush=True)
    print(f"Saved all available results: {all_csv}", flush=True)

    if errors:
        print("ERROR: Grid incomplete; no final best-by-validation summary written", flush=True)
        for message in errors[:20]:
            print(" -", message, flush=True)
        if len(errors) > 20:
            print(f" ... and {len(errors) - 20} more", flush=True)
        sys.exit(1)

    write_csv(best_csv, selected)
    print(f"\nFINAL VALIDATION-SELECTED SUMMARY: {dataset}", flush=True)
    for model in EXPECTED_PER_HORIZON:
        print(f"\n{model}", flush=True)
        for r in selected:
            if r["model"] == model:
                print(
                    f"H={r['horizon']:<3} L={r['lookback']:<3} "
                    f"Val MSE={r['validation_mse']:.7f} "
                    f"Test MSE={r['test_mse']:.7f} "
                    f"Test MAE={r['test_mae']:.7f} "
                    f"Best epoch={r['best_epoch']} Params={r['hyperparameters']}",
                    flush=True,
                )
    print(f"\nSaved: {best_csv}", flush=True)
    print("Selection criterion: lowest per-epoch validation MSE (not test MSE)", flush=True)


if __name__ == "__main__":
    main()
