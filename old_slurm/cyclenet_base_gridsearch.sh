
#!/usr/bin/env bash

# ============================================================
# BASE CYCLENET GRID SEARCH
# Datasets: Solar, Weather, ETTm1
# Lookbacks: 96, 192, 336, 720
# Horizons: 96, 192, 336, 720
# Total: 48 experiments
#
# MODEL SELECTION:
#   Lowest validation MSE per dataset and horizon.
# REPORT:
#   Test MSE and Test MAE of the selected configuration.
#
# Completed runs are skipped on resubmission.
# ============================================================

set -euo pipefail

cd /home/alalade/CycleNet

RESULTS_DIR="./results/baseline/cyclenet_baseline_results"
mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/done"

# ------------------------------------------------------------
# Parse completed experiments and automatically select winners
# ------------------------------------------------------------

update_summaries() {

  python - "$RESULTS_DIR" "${1:-}" <<'PY'
import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
new_id = sys.argv[2]

number = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

model_re = re.compile(
    r"^(Solar|Weather|ETTm1)_CycleNet_"
    r"L(96|192|336|720)_H(96|192|336|720)_"
    r"lr(\d+(?:\.\d+)?)$"
)


def parse_run(run_id):

    match = model_re.fullmatch(run_id)

    if not match:
        raise ValueError(f"Invalid experiment ID: {run_id}")

    path = root / "logs" / f"{run_id}.log"
    text = path.read_text(errors="replace")

    epochs = [
        (int(epoch), float(val))
        for epoch, val in re.findall(
            rf"Epoch:\s*(\d+),\s*Steps:.*?"
            rf"Val MSE:\s*({number})",
            text
        )
    ]

    summaries = [
        line for line in text.splitlines()
        if line.startswith("VALIDATION RESULT |")
    ]

    if summaries:

        line = summaries[-1]

        fields = dict(re.findall(
            rf"(best_epoch|best_val_mse|test_mse|test_mae)"
            rf"=({number})",
            line
        ))

        if len(fields) != 4:
            raise ValueError(
                f"Incomplete validation summary: {run_id}"
            )

        best_epoch = int(float(fields["best_epoch"]))
        val_mse = float(fields["best_val_mse"])
        test_mse = float(fields["test_mse"])
        test_mae = float(fields["test_mae"])

        if epochs:
            minimum = min(v for _, v in epochs)

            if abs(val_mse - minimum) > 0.000002:
                raise ValueError(
                    f"Validation MSE mismatch: {run_id}"
                )

    else:

        if not epochs:
            raise ValueError(
                f"No validation MSE found: {run_id}"
            )

        best_epoch, val_mse = min(
            epochs, key=lambda x: x[1]
        )

        # Support detailed test output and standard CycleNet output.
        blocks = text.split("FINAL TEST RESULTS")

        if len(blocks) > 1:

            block = blocks[-1]

            mse = re.search(
                rf"(?m)^\s*MSE:\s*({number})\s*$",
                block
            )

            mae = re.search(
                rf"(?m)^\s*MAE:\s*({number})\s*$",
                block
            )

            if not (mse and mae):
                raise ValueError(
                    f"Incomplete test results: {run_id}"
                )

            test_mse = float(mse.group(1))
            test_mae = float(mae.group(1))

        else:

            pairs = re.findall(
                rf"\bmse\s*:\s*({number})"
                rf"\s*,\s*mae\s*:\s*({number})",
                text,
                re.I
            )

            if not pairs:
                raise ValueError(
                    f"No test MSE/MAE found: {run_id}"
                )

            test_mse, test_mae = map(float, pairs[-1])

    return {
        "dataset": match.group(1),
        "horizon": int(match.group(3)),
        "lookback": int(match.group(2)),
        "learning_rate": match.group(4),
        "best_epoch": best_epoch,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "experiment_id": run_id
    }


# Verify a newly completed experiment before marking it done.
if new_id:

    result = parse_run(new_id)

    (root / "done" / f"{new_id}.done").touch()

    print(
        f"VERIFIED: {new_id} | "
        f"Val MSE={result['val_mse']:.7f} | "
        f"Test MSE={result['test_mse']:.7f} | "
        f"Test MAE={result['test_mae']:.7f}",
        flush=True
    )


# Recover all previously completed experiments.
rows = [
    parse_run(path.stem)
    for path in sorted((root / "done").glob("*.done"))
]


def write_csv(filename, columns, data):

    with (root / filename).open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(data)


columns = [
    "dataset",
    "horizon",
    "lookback",
    "learning_rate",
    "best_epoch",
    "val_mse",
    "test_mse",
    "test_mae",
    "experiment_id"
]

write_csv("all_completed_results.csv", columns, rows)


# Select the lowest validation MSE, never the lowest test MSE.
best = {}

for row in rows:

    key = (row["dataset"], row["horizon"])

    if key not in best or (
        row["val_mse"], row["lookback"]
    ) < (
        best[key]["val_mse"], best[key]["lookback"]
    ):
        best[key] = row


winners = [best[key] for key in sorted(best)]

write_csv("best_by_val_mse.csv", columns, winners)


# Create separate test MSE and test MAE comparison tables.
for metric in ("test_mse", "test_mae"):

    comparison = []

    for horizon in (96, 192, 336, 720):

        line = {"horizon": horizon}

        for dataset in ("Solar", "Weather", "ETTm1"):

            key = (dataset, horizon)

            line[dataset] = (
                f"{best[key][metric]:.7f}"
                if key in best else ""
            )

        comparison.append(line)

    write_csv(
        f"{metric}_comparison.csv",
        ["horizon", "Solar", "Weather", "ETTm1"],
        comparison
    )


print(
    f"SUMMARY UPDATED: {len(rows)}/48 validated runs, "
    f"{len(winners)}/12 winners",
    flush=True
)


if len(rows) == 48:

    print("\nFINAL WINNERS (LOWEST VALIDATION MSE)")

    print(
        f"{'Dataset':<9} {'H':>4} {'L':>4} "
        f"{'Val MSE':>12} {'Test MSE':>12} "
        f"{'Test MAE':>12}"
    )

    for row in winners:

        print(
            f"{row['dataset']:<9} "
            f"{row['horizon']:>4} "
            f"{row['lookback']:>4} "
            f"{row['val_mse']:>12.7f} "
            f"{row['test_mse']:>12.7f} "
            f"{row['test_mae']:>12.7f}"
        )
PY
}

# Rebuild summaries from completed runs after resubmission.
update_summaries


# ------------------------------------------------------------
# Dataset configurations
# ------------------------------------------------------------

datasets=(
  "solar_AL.txt|Solar|Solar|137|144"
  "weather.csv|custom|Weather|21|144"
  "ETTm1.csv|ETTm1|ETTm1|7|96"
)

trial=0

# ------------------------------------------------------------
# Execute experiments
# ------------------------------------------------------------

for entry in "${datasets[@]}"; do

  IFS='|' read -r data_path data dataset enc_in cycle <<< "$entry"

  if [[ ! -f "./dataset/$data_path" ]]; then
    echo "Missing dataset: ./dataset/$data_path" >&2
    exit 1
  fi

  for horizon in 96 192 336 720; do

    for lookback in 96 192 336 720; do

      ((trial += 1))

      id="${dataset}_CycleNet_L${lookback}_H${horizon}_lr0.0001"

      log="$RESULTS_DIR/logs/$id.log"
      done_marker="$RESULTS_DIR/done/$id.done"

      if [[ -f "$done_marker" ]]; then
        echo "[$trial/48] SKIPPING: $id"
        continue
      fi

      echo "=============================================="
      echo "RUN [$trial/48]: $id"
      echo "=============================================="

      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "$id" \
        --model CycleNet \
        --data "$data" \
        --features M \
        --seq_len "$lookback" \
        --pred_len "$horizon" \
        --enc_in "$enc_in" \
        --cycle "$cycle" \
        --model_type mlp \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --lradj type3 \
        --train_epochs 30 \
        --patience 5 \
        --random_seed 2024 \
        --itr 1 \
        2>&1 | tee "$log"

      # Validate, save results and update winning configurations.
      update_summaries "$id"

    done
  done
done

echo "=============================================="
echo "ALL 48 EXPERIMENTS COMPLETED"
echo "Results directory: $RESULTS_DIR"
echo "=============================================="
