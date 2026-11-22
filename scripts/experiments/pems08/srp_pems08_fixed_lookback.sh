#!/usr/bin/env bash
#SBATCH --job-name=srp_pems08
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=srp_pems08_%j.out
#SBATCH --error=srp_pems08_%j.err

# Fixed-lookback CycleNet mixing grid: PEMS08.
# Models: CycleNetMix, PairMix, MixLowRank.
# Each (model, lookback, horizon) selects the lowest validation MSE.
# Report test MSE and MAE from that same winning trial.
# 1408 trials total.
# Resumable: completed trials are skipped when resubmitted.

set -Eeuo pipefail

PROJECT_DIR="/home/alalade/CycleNet"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"

which python
python --version

DATASET="PEMS08"
DATA_LOADER="PEMS"
DATA_FILE="PEMS08.npz"
ENC_IN=170
CYCLE=288
REVIN=1

RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/fixed_lookback_grid/pems08}"
MODE="${1:---run}"
SEED="${SEED:-2024}"

read -r -a LOOKBACKS <<< "${LOOKBACKS_OVERRIDE:-96 192 336 720}"
read -r -a HORIZONS <<< "${HORIZONS_OVERRIDE:-96 192 336 720}"
read -r -a LEARNING_RATES <<< "${LEARNING_RATES_OVERRIDE:-0.0001 0.0005 0.001 0.002}"
read -r -a BATCH_SIZES <<< "${BATCH_SIZES_OVERRIDE:-32 64}"
read -r -a MIX_LAYERS <<< "${MIX_LAYERS_OVERRIDE:-1 2}"
read -r -a MIX_RATIOS <<< "${MIX_RATIOS_OVERRIDE:-0.2 0.4 0.6}"
read -r -a MIX_RANKS <<< "${MIX_RANKS_OVERRIDE:-16 32 64}"

MIX_HIDDEN=64
LOWRANK_HIDDEN=128
MIX_DROPOUT=0.1
MIX_LAG_WINDOW=4
TRAIN_EPOCHS=30
PATIENCE=5

# Dependency matrix generated from pretrained CycleNet residuals.
# Baseline: lookback 336, horizon 96, cycle 288, RevIN 1.
PAIR_MATRIX="${PAIR_MATRIX:-${PROJECT_DIR}/dependency_scores/PEMS08_dependency.npy}"

mkdir -p \
    "$RESULTS_DIR/logs" \
    "$RESULTS_DIR/records" \
    "$RESULTS_DIR/checkpoints"

MANIFEST="$RESULTS_DIR/manifest.csv"

if [[ "$MODE" != "--run" &&
      "$MODE" != "--plan" &&
      "$MODE" != "--summarize-only" ]]; then
    echo "Usage: bash $0 [--run | --plan | --summarize-only]"
    exit 2
fi

# ------------------------------------------------------------
# 1. Experiment naming
# ------------------------------------------------------------

make_id() {
    local variant="$1"
    local L="$2"
    local H="$3"
    local bs="$4"
    local lr="$5"
    local extra="$6"

    local lr_slug="${lr//./p}"

    printf '%s_%s_L%s_H%s_bs%s_lr%s_%s_seed%s' \
        "$DATASET" "$variant" "$L" "$H" \
        "$bs" "$lr_slug" "$extra" "$SEED"
}

# ------------------------------------------------------------
# 2. Construct hyperparameter search manifest
# ------------------------------------------------------------

write_manifest() {
    local temp="${MANIFEST}.tmp.$$"

    printf '%s\n' \
'experiment_id,dataset,variant,lookback,horizon,learning_rate,batch_size,mix_layers,mix_ratio,mix_rank,mix_hidden,mix_dropout,seed' \
        > "$temp"

    local L H lr bs layers ratio rank id

    for L in "${LOOKBACKS[@]}"; do
        for H in "${HORIZONS[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do
                for lr in "${LEARNING_RATES[@]}"; do

                    # CycleNetMix
                    for layers in "${MIX_LAYERS[@]}"; do
                        id="$(make_id \
                            CycleNetMix "$L" "$H" "$bs" "$lr" \
                            "layers${layers}")"

                        printf '%s,%s,%s,%s,%s,%s,%s,%s,,,%s,%s,%s\n' \
                            "$id" "$DATASET" CycleNetMix \
                            "$L" "$H" "$lr" "$bs" "$layers" \
                            "$MIX_HIDDEN" "$MIX_DROPOUT" "$SEED" \
                            >> "$temp"
                    done

                    # PairMix
                    for ratio in "${MIX_RATIOS[@]}"; do
                        id="$(make_id \
                            PairMix "$L" "$H" "$bs" "$lr" \
                            "mr${ratio//./p}")"

                        printf '%s,%s,%s,%s,%s,%s,%s,,%s,,128,%s,%s\n' \
                            "$id" "$DATASET" PairMix \
                            "$L" "$H" "$lr" "$bs" "$ratio" \
                            "$MIX_DROPOUT" "$SEED" \
                            >> "$temp"
                    done

                    # MixLowRank
                    for rank in "${MIX_RANKS[@]}"; do
                        for layers in "${MIX_LAYERS[@]}"; do
                            id="$(make_id \
                                MixLowRank "$L" "$H" "$bs" "$lr" \
                                "rank${rank}_layers${layers}")"

                            printf '%s,%s,%s,%s,%s,%s,%s,%s,,%s,%s,%s,%s\n' \
                                "$id" "$DATASET" MixLowRank \
                                "$L" "$H" "$lr" "$bs" "$layers" \
                                "$rank" "$LOWRANK_HIDDEN" \
                                "$MIX_DROPOUT" "$SEED" \
                                >> "$temp"
                        done
                    done

                done
            done
        done
    done

    # Prevent mixing experiment results from different grids.
    if [[ -f "$MANIFEST" ]] &&
       find "$RESULTS_DIR/records" -maxdepth 1 \
            -name '*.json' -print -quit | grep -q .; then

        if ! cmp -s "$temp" "$MANIFEST"; then
            echo "ERROR: Existing completed records belong to a different grid."
            echo "Use a new RESULTS_DIR if you change the search or seed."
            rm -f "$temp"
            exit 1
        fi
    fi

    mv "$temp" "$MANIFEST"
}

# ------------------------------------------------------------
# 3. Embedded results summarizer
# ------------------------------------------------------------

summarize() {
    local new_id="${1:-}"
    local recovery="${2:-no}"

    python - "$RESULTS_DIR" "$new_id" "$recovery" <<'PY_SUMMARY'
import csv
import json
import math
import os
import re
import sys

from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
new_id = sys.argv[2]
recover = sys.argv[3] == "yes"

num = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

with (root / "manifest.csv").open(newline="") as f:
    planned = list(csv.DictReader(f))

ids = {r["experiment_id"]: r for r in planned}

if len(ids) != len(planned):
    raise ValueError("Duplicate experiment IDs")


def parse_trial(run_id):
    path = root / "logs" / (run_id + ".log")
    content = path.read_text(errors="replace")

    final = content.rsplit("FINAL TEST RESULTS", 1)

    if len(final) != 2:
        raise ValueError("No FINAL TEST RESULTS block")

    tmse = re.search(
        rf"(?im)^\s*MSE\s*:\s*({num})\s*$",
        final[-1]
    )

    tmae = re.search(
        rf"(?im)^\s*MAE\s*:\s*({num})\s*$",
        final[-1]
    )

    if not (tmse and tmae):
        raise ValueError("Missing final test metrics")

    epoch_metrics = [
        (int(e), float(m))
        for e, m in re.findall(
            rf"(?m)^\s*Epoch:\s*(\d+),\s*Steps:.*?Val\s+MSE:\s*({num})",
            content
        )
    ]

    if not epoch_metrics:
        raise ValueError("Missing training validation MSE")

    best_epoch, best_val = min(
        epoch_metrics,
        key=lambda z: (z[1], z[0])
    )

    summaries = [
        line for line in content.splitlines()
        if line.startswith("VALIDATION RESULT |")
    ]

    if summaries:
        summary = summaries[-1]

        match = re.search(
            rf"best_val_mse=({num})",
            summary
        )

        epoch = re.search(
            r"best_epoch=(\d+)",
            summary
        )

        if not (match and epoch):
            raise ValueError("Incomplete validation summary")

        given_val = float(match.group(1))

        if abs(given_val - best_val) > 2e-6:
            raise ValueError(
                "Validation summary differs from epoch minimum"
            )

        best_val = given_val
        best_epoch = int(epoch.group(1))

    mse = float(tmse.group(1))
    mae = float(tmae.group(1))

    if not all(map(math.isfinite, (best_val, mse, mae))):
        raise ValueError("Non-finite metric")

    return {
        "best_epoch": best_epoch,
        "val_mse": best_val,
        "test_mse": mse,
        "test_mae": mae
    }


def atomic_json(path, item):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(item, indent=2) + "\n")
    os.replace(temp, path)


# Recover completed experiments that lack JSON records.
if recover:
    for run_id in ids:
        record = root / "records" / (run_id + ".json")
        log = root / "logs" / (run_id + ".log")

        if not record.exists() and log.exists():
            try:
                result = parse_trial(run_id)
            except (ValueError, OSError) as exc:
                print(
                    f"INCOMPLETE (will rerun): {run_id}: {exc}",
                    flush=True
                )
            else:
                atomic_json(record, result)
                print(f"RECOVERED: {run_id}", flush=True)


if new_id:
    if new_id not in ids:
        raise ValueError("Trial absent from manifest")

    result = parse_trial(new_id)

    atomic_json(
        root / "records" / (new_id + ".json"),
        result
    )

    print(
        f'VERIFIED: {new_id} '
        f'| Val MSE={result["val_mse"]:.7f} '
        f'| Test MSE={result["test_mse"]:.7f} '
        f'| Test MAE={result["test_mae"]:.7f}',
        flush=True
    )


rows = []

for meta in planned:
    record = root / "records" / (
        meta["experiment_id"] + ".json"
    )

    if record.exists():
        metrics = json.loads(record.read_text())

        row = {**meta, **metrics}

        row["lookback"] = int(row["lookback"])
        row["horizon"] = int(row["horizon"])
        row["batch_size"] = int(row["batch_size"])

        rows.append(row)


columns = (
    list(planned[0])
    + ["best_epoch", "val_mse", "test_mse", "test_mae"]
)


def export(name, data, fields=None):
    fields = fields or columns
    dest = root / name
    temp = dest.with_suffix(dest.suffix + ".tmp")

    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(data)

    os.replace(temp, dest)


export("all_completed_trials.csv", rows)

expected = defaultdict(set)
groups = defaultdict(list)

for entry in planned:
    key = (
        entry["variant"],
        int(entry["lookback"]),
        int(entry["horizon"])
    )
    expected[key].add(entry["experiment_id"])

for entry in rows:
    key = (
        entry["variant"],
        entry["lookback"],
        entry["horizon"]
    )
    groups[key].append(entry)


selected = []

for key in sorted(
    groups,
    key=lambda k: (k[1], k[2], k[0])
):
    variants = groups[key]

    winner = min(
        variants,
        key=lambda r: (
            r["val_mse"],
            r["experiment_id"]
        )
    )

    done_count = len({
        r["experiment_id"] for r in variants
    })

    full_count = len(expected[key])

    selected.append({
        **winner,
        "trials_finished": done_count,
        "trials_planned": full_count,
        "status": (
            "COMPLETE"
            if done_count == full_count
            else "PROVISIONAL"
        )
    })


win_cols = columns + [
    "trials_finished",
    "trials_planned",
    "status"
]

export(
    "best_by_val_mse_so_far.csv",
    selected,
    win_cols
)

final = [
    r for r in selected
    if r["status"] == "COMPLETE"
]

export(
    "best_by_val_mse_complete.csv",
    final,
    win_cols
)


# Wide comparison table
wide_fields = ["dataset", "lookback", "horizon"]

for variant in [
    "CycleNetMix",
    "PairMix",
    "MixLowRank"
]:
    wide_fields += [
        f"{variant}_val_mse",
        f"{variant}_test_mse",
        f"{variant}_test_mae",
        f"{variant}_status"
    ]

lookup = {
    (r["lookback"], r["horizon"], r["variant"]): r
    for r in selected
}

wide = []

for L, H in sorted({
    (int(x["lookback"]), int(x["horizon"]))
    for x in planned
}):
    item = {
        "dataset": planned[0]["dataset"],
        "lookback": L,
        "horizon": H
    }

    for variant in [
        "CycleNetMix",
        "PairMix",
        "MixLowRank"
    ]:
        v = lookup.get((L, H, variant))

        if v:
            item.update({
                f"{variant}_val_mse": v["val_mse"],
                f"{variant}_test_mse": v["test_mse"],
                f"{variant}_test_mae": v["test_mae"],
                f"{variant}_status": v["status"]
            })

    wide.append(item)


export(
    "test_comparison_by_fixed_lookback.csv",
    wide,
    wide_fields
)

print(
    f"SUMMARY: {len(rows)}/{len(planned)} trials complete; "
    f"{len(final)}/{len(expected)} groups finalized",
    flush=True
)

if len(rows) == len(planned):
    print("ALL TRIALS COMPLETE. Final winners:")

    for x in final:
        print(
            f"  {x['variant']:<13} "
            f"L={x['lookback']:<3} "
            f"H={x['horizon']:<3} "
            f"val={x['val_mse']:.6f} "
            f"test MSE={x['test_mse']:.6f} "
            f"test MAE={x['test_mae']:.6f}"
        )
PY_SUMMARY
}

# ------------------------------------------------------------
# 4. Prepare experiment
# ------------------------------------------------------------

write_manifest

TOTAL=$(( $(wc -l < "$MANIFEST") - 1 ))

echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Lookbacks: ${LOOKBACKS[*]}"
echo "Horizons: ${HORIZONS[*]}"
echo "Learning rates: ${LEARNING_RATES[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Models: CycleNetMix, PairMix, MixLowRank"
echo "Planned trials: $TOTAL"
echo "Results: $RESULTS_DIR"

if [[ "$MODE" == "--plan" ]]; then
    echo "Manifest generated. No training started."
    exit 0
fi

summarize "" yes

if [[ "$MODE" == "--summarize-only" ]]; then
    exit 0
fi

# ------------------------------------------------------------
# 5. Validate dataset
# ------------------------------------------------------------

if [[ ! -f "./run.py" ]]; then
    echo "ERROR: run.py not found."
    exit 1
fi

if [[ ! -f "./dataset/$DATA_FILE" ]]; then
    echo "ERROR: PEMS08 dataset not found."
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

command -v "$PYTHON_BIN" >/dev/null || {
    echo "ERROR: Python unavailable"
    exit 1
}

"$PYTHON_BIN" - "$PROJECT_DIR/dataset/$DATA_FILE" "$ENC_IN" <<'PY_DATA'
import numpy as np
import sys

path = sys.argv[1]
expected = int(sys.argv[2])

with np.load(path, allow_pickle=False) as archive:
    if "data" not in archive.files:
        raise SystemExit("PEMS archive missing key 'data'")

    shape = archive["data"].shape

if len(shape) not in (2, 3) or shape[1] != expected:
    raise SystemExit(
        f"Unexpected PEMS08 shape: {shape}"
    )

print("PEMS08 raw data shape:", shape)
print("Expected model channels:", expected)
print("Confirm Dataset_PEMS uses 170 channels.")
PY_DATA

# ------------------------------------------------------------
# 6. Validate dependency matrix
# ------------------------------------------------------------

if [[ ! -f "$PAIR_MATRIX" ]]; then
    echo "ERROR: Dependency matrix missing: $PAIR_MATRIX"
    exit 1
fi

"$PYTHON_BIN" - "$PAIR_MATRIX" "$ENC_IN" <<'PY_MATRIX'
import numpy as np
import sys

path = sys.argv[1]
expected = int(sys.argv[2])

D = np.load(path, allow_pickle=False)

if D.shape != (expected, expected):
    raise SystemExit(
        f"Invalid dependency shape: {D.shape}"
    )

if not np.isfinite(D).all():
    raise SystemExit("NaN/Inf in dependency matrix")

if not np.allclose(D, D.T, atol=1e-5):
    raise SystemExit("Dependency matrix not symmetric")

print(
    "PairMix dependency matrix verified:",
    path,
    D.shape
)
PY_MATRIX

# ------------------------------------------------------------
# 7. Run experiments
# ------------------------------------------------------------

TRIAL_NUM=0

run_one() {
    local variant="$1"
    local L="$2"
    local H="$3"
    local bs="$4"
    local lr="$5"
    local layers="$6"
    local ratio="$7"
    local rank="$8"

    local extra id model
    local -a extra_flags=()

    case "$variant" in
        CycleNetMix)
            model="CycleNetMixTune"
            extra="layers${layers}"

            extra_flags=(
                --mix_hidden "$MIX_HIDDEN"
                --mix_layers "$layers"
                --mix_dropout "$MIX_DROPOUT"
            )
            ;;

        PairMix)
            model="CycleNetPairMix"
            extra="mr${ratio//./p}"

            extra_flags=(
                --mix_ratio "$ratio"
                --dependency_score_path "$PAIR_MATRIX"
                --mix_lag_window "$MIX_LAG_WINDOW"
                --mix_dropout "$MIX_DROPOUT"
            )
            ;;

        MixLowRank)
            model="CycleNetMixLowRank"
            extra="rank${rank}_layers${layers}"

            extra_flags=(
                --mix_rank "$rank"
                --mix_layers "$layers"
                --mix_hidden "$LOWRANK_HIDDEN"
                --mix_dropout "$MIX_DROPOUT"
            )
            ;;

        *)
            echo "Unknown model: $variant"
            exit 2
            ;;
    esac

    id="$(make_id \
        "$variant" "$L" "$H" "$bs" "$lr" "$extra")"

    ((TRIAL_NUM += 1))

    if [[ -f "$RESULTS_DIR/records/$id.json" ]]; then
        echo "[$TRIAL_NUM/$TOTAL] SKIP $id"
        return
    fi

    echo "[$TRIAL_NUM/$TOTAL] TRAIN $id"

    local log="$RESULTS_DIR/logs/$id.log"
    local checkpoints="$RESULTS_DIR/checkpoints/$id"

    mkdir -p "$checkpoints"

    "$PYTHON_BIN" -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$DATA_FILE" \
        --model_id "$id" \
        --model "$model" \
        --data "$DATA_LOADER" \
        --features M \
        --seq_len "$L" \
        --pred_len "$H" \
        --enc_in "$ENC_IN" \
        --cycle "$CYCLE" \
        --model_type mlp \
        --use_revin "$REVIN" \
        --train_epochs "$TRAIN_EPOCHS" \
        --patience "$PATIENCE" \
        --itr 1 \
        --batch_size "$bs" \
        --learning_rate "$lr" \
        --lradj type3 \
        --num_workers 4 \
        --random_seed "$SEED" \
        --checkpoints "$checkpoints/" \
        "${extra_flags[@]}" \
        2>&1 | tee "$log"

    summarize "$id" no
}

for L in "${LOOKBACKS[@]}"; do
    for H in "${HORIZONS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do

                for layers in "${MIX_LAYERS[@]}"; do
                    run_one CycleNetMix \
                        "$L" "$H" "$bs" "$lr" \
                        "$layers" "" ""
                done

                for ratio in "${MIX_RATIOS[@]}"; do
                    run_one PairMix \
                        "$L" "$H" "$bs" "$lr" \
                        "" "$ratio" ""
                done

                for rank in "${MIX_RANKS[@]}"; do
                    for layers in "${MIX_LAYERS[@]}"; do
                        run_one MixLowRank \
                            "$L" "$H" "$bs" "$lr" \
                            "$layers" "" "$rank"
                    done
                done

            done
        done
    done
done

echo "COMPLETE: All $TOTAL PEMS08 experiments finished."
echo "Winners: $RESULTS_DIR/best_by_val_mse_complete.csv"
echo "Test comparison: $RESULTS_DIR/test_comparison_by_fixed_lookback.csv"
