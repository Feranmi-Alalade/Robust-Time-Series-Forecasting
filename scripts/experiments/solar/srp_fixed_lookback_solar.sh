#!/usr/bin/env bash
#SBATCH --job-name=srp_solar
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=srp_solar_%j.out
#SBATCH --error=srp_solar_%j.err

# Fixed-lookback CycleNet mixing grid: Solar.
# Exactly ONE file is needed for this dataset (Python summarizer is embedded).
# Models: CycleNetMixTune, CycleNetPairMix, CycleNetMixLowRank.
# Each (model, lookback, horizon) independently selects the lowest training-time
# validation MSE; reports TEST MSE and MAE from that SAME selected trial.
# Batch-size tuning doubles the original 704-trial grid: 1,408 runs per dataset.
# RESUMABLE: resubmit the same file with sbatch; completed trials are skipped.
# --plan and --summarize-only can also be run with bash (outside SLURM).
set -Eeuo pipefail

# Activate Python environment before any Python commands
cd /home/alalade/CycleNet
source /home/alalade/CycleNet/venv/bin/activate

# Verify Python is available
which python
python --version

DATASET="Solar"
DATA_LOADER="Solar"
DATA_FILE="solar_AL.txt"
ENC_IN=137
CYCLE=144
REVIN=0
PROJECT_DIR="${PROJECT_DIR:-/home/alalade/CycleNet}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/fixed_lookback_grid/solar}"
MODE="${1:---run}"
SEED="${SEED:-2024}"

# Each L is a separate group, NOT a hyperparameter to select across groups.
# Override search arrays via environment variables if a smaller grid is desired.
read -r -a LOOKBACKS      <<< "${LOOKBACKS_OVERRIDE:-96 192 336 720}"
read -r -a HORIZONS       <<< "${HORIZONS_OVERRIDE:-96 192 336 720}"
read -r -a LEARNING_RATES <<< "${LEARNING_RATES_OVERRIDE:-0.0001 0.0005 0.001 0.002}"
read -r -a BATCH_SIZES    <<< "${BATCH_SIZES_OVERRIDE:-32 64}"
read -r -a MIX_LAYERS     <<< "${MIX_LAYERS_OVERRIDE:-1 2}"
read -r -a MIX_RATIOS     <<< "${MIX_RATIOS_OVERRIDE:-0.2 0.4 0.6}"
read -r -a MIX_RANKS      <<< "${MIX_RANKS_OVERRIDE:-16 32 64}"
# Additional knobs can be extended deliberately, with a larger compute cost.
# These settings match the previous experiment's model definitions.
MIX_HIDDEN=64
LOWRANK_HIDDEN=128
MIX_DROPOUT=0.1
MIX_LAG_WINDOW=4
TRAIN_EPOCHS=30
PATIENCE=5

mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/records" "$RESULTS_DIR/checkpoints"
MANIFEST="$RESULTS_DIR/manifest.csv"

if [[ "$MODE" != '--run' && "$MODE" != '--plan' && "$MODE" != '--summarize-only' ]]; then
  echo 'Usage: bash SCRIPT.sh [--run | --plan | --summarize-only]' >&2
  exit 2
fi

# Fields: experiment_id,dataset,variant,lookback,horizon,learning_rate,batch_size,
#         mix_layers,mix_ratio,mix_rank,mix_hidden,mix_dropout,seed
# Stable IDs include EVERY tuned dimension and the seed, preventing overwrite.
make_id() {
  local variant="$1" L="$2" H="$3" bs="$4" lr="$5" extra="$6"
  local lr_slug="${lr//./p}"
  printf '%s_%s_L%s_H%s_bs%s_lr%s_%s_seed%s' "$DATASET" "$variant" "$L" "$H" "$bs" "$lr_slug" "$extra" "$SEED"
}

# Builds deterministic search manifest and prevents mixing distinct L values.
write_manifest() {
  local temp="${MANIFEST}.tmp.$$"
  printf '%s\n' 'experiment_id,dataset,variant,lookback,horizon,learning_rate,batch_size,mix_layers,mix_ratio,mix_rank,mix_hidden,mix_dropout,seed' > "$temp"
  local L H lr bs layers ratio rank id
  for L in "${LOOKBACKS[@]}"; do
    for H in "${HORIZONS[@]}"; do
      for bs in "${BATCH_SIZES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
          for layers in "${MIX_LAYERS[@]}"; do
            id="$(make_id CycleNetMix "$L" "$H" "$bs" "$lr" "layers${layers}")"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,,,%s,%s,%s\n' "$id" "$DATASET" 'CycleNetMix' "$L" "$H" "$lr" "$bs" "$layers" "$MIX_HIDDEN" "$MIX_DROPOUT" "$SEED" >> "$temp"
          done
          for ratio in "${MIX_RATIOS[@]}"; do
            id="$(make_id PairMix "$L" "$H" "$bs" "$lr" "mr${ratio//./p}")"
            printf '%s,%s,%s,%s,%s,%s,%s,,%s,,128,%s,%s\n' "$id" "$DATASET" 'PairMix' "$L" "$H" "$lr" "$bs" "$ratio" "$MIX_DROPOUT" "$SEED" >> "$temp"
          done
          for rank in "${MIX_RANKS[@]}"; do
            for layers in "${MIX_LAYERS[@]}"; do
              id="$(make_id MixLowRank "$L" "$H" "$bs" "$lr" "rank${rank}_layers${layers}")"
              printf '%s,%s,%s,%s,%s,%s,%s,%s,,%s,%s,%s,%s\n' "$id" "$DATASET" 'MixLowRank' "$L" "$H" "$lr" "$bs" "$layers" "$rank" "$LOWRANK_HIDDEN" "$MIX_DROPOUT" "$SEED" >> "$temp"
            done
          done
        done
      done
    done
  done
  mv "$temp" "$MANIFEST"
}

# Parse completed trial logs, then save a CSV per each fixed-L/H winner.
# Runs without complete metrics are never marked finished.
# The VALIDATION RESULT's best_val_mse (or minimum epoch Val MSE) determines
# configuration selection. The later BEST VALIDATION CHECKPOINT re-evaluation
# can differ slightly and is NOT used to rank configurations.
summarize() {
  local new_id="${1:-}" recovery="${2:-no}"
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
recover = sys.argv[3] == 'yes'
num = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
manifest_path = root / 'manifest.csv'
with manifest_path.open(newline='') as handle:
    planned = list(csv.DictReader(handle))
ids = {r['experiment_id']: r for r in planned}
if len(ids) != len(planned):
    raise ValueError('Duplicate experiment IDs in search manifest')

def parse_trial(run_id):
    path = root / 'logs' / (run_id + '.log')
    content = path.read_text(errors='replace')
    # A successful full training run contains a distinct FINAL TEST RESULTS block.
    final = content.rsplit('FINAL TEST RESULTS', 1)
    if len(final) != 2:
        raise ValueError('No FINAL TEST RESULTS block')
    tmse = re.search(rf'(?im)^\s*MSE\s*:\s*({num})\s*$', final[-1])
    tmae = re.search(rf'(?im)^\s*MAE\s*:\s*({num})\s*$', final[-1])
    if not (tmse and tmae):
        raise ValueError('Missing final TEST MSE or TEST MAE')
    epoch_metrics = [
        (int(e), float(m)) for e, m in re.findall(
            rf'(?m)^\s*Epoch:\s*(\d+),\s*Steps:.*?Val\s+MSE:\s*({num})', content
        )
    ]
    if not epoch_metrics:
        raise ValueError('Missing training-time epoch validation MSE')
    best_epoch, best_val = min(epoch_metrics, key=lambda z: (z[1], z[0]))
    summary_lines = [x for x in content.splitlines() if x.startswith('VALIDATION RESULT |')]
    if summary_lines:
        summary = summary_lines[-1]
        match = re.search(rf'best_val_mse=({num})', summary)
        epoch = re.search(r'best_epoch=(\d+)', summary)
        if not (match and epoch):
            raise ValueError('Incomplete VALIDATION RESULT line')
        given_val = float(match.group(1))
        if abs(given_val - best_val) > 2e-6:
            raise ValueError(f'VALIDATION RESULT {given_val} differs from minimum epoch MSE {best_val}')
        best_val = given_val
        best_epoch = int(epoch.group(1))
    mse, mae = float(tmse.group(1)), float(tmae.group(1))
    if not all(map(math.isfinite, (best_val, mse, mae))):
        raise ValueError('Non-finite metric')
    return dict(best_epoch=best_epoch, val_mse=best_val, test_mse=mse, test_mae=mae)

def atomic_json(path, item):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(item, indent=2) + '\n')
    os.replace(temp, path)

# Recover a completed log whose job was stopped before recording its success.
if recover:
    for run_id in ids:
        record = root / 'records' / (run_id + '.json')
        log = root / 'logs' / (run_id + '.log')
        if not record.exists() and log.exists():
            try:
                result = parse_trial(run_id)
            except (ValueError, OSError) as exc:
                print(f'INCOMPLETE (will rerun): {run_id}: {exc}', flush=True)
            else:
                atomic_json(record, result)
                print(f'RECOVERED: {run_id}', flush=True)

if new_id:
    if new_id not in ids:
        raise ValueError('New trial absent from search manifest: ' + new_id)
    result = parse_trial(new_id)  # Fail closed: never mark a failed run complete.
    atomic_json(root / 'records' / (new_id + '.json'), result)
    print(f'VERIFIED: {new_id} | Val MSE={result["val_mse"]:.7f}'
          f' | Test MSE={result["test_mse"]:.7f} | Test MAE={result["test_mae"]:.7f}', flush=True)

rows = []
for meta in planned:
    record = root / 'records' / (meta['experiment_id'] + '.json')
    if record.exists():
        metrics = json.loads(record.read_text())
        row = {**meta, **metrics}
        row['lookback'] = int(row['lookback'])
        row['horizon'] = int(row['horizon'])
        row['batch_size'] = int(row['batch_size'])
        rows.append(row)

columns = list(planned[0]) + ['best_epoch', 'val_mse', 'test_mse', 'test_mae'] if planned else []

def export(name, data, fields=None):
    fields = fields or columns
    dest = root / name
    temp = dest.with_suffix(dest.suffix + '.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)
    os.replace(temp, dest)

export('all_completed_trials.csv', rows)
completed = {x['experiment_id'] for x in rows}
expected = defaultdict(set)
groups = defaultdict(list)
for entry in planned:
    expected[(entry['variant'], int(entry['lookback']), int(entry['horizon']))].add(entry['experiment_id'])
for entry in rows:
    groups[(entry['variant'], entry['lookback'], entry['horizon'])].append(entry)

selected = []
for key in sorted(groups, key=lambda k: (k[1], k[2], k[0])):
    variants = groups[key]
    winner = min(variants, key=lambda r: (r['val_mse'], r['experiment_id']))
    done_count = len({r['experiment_id'] for r in variants})
    full_count = len(expected[key])
    selected.append({**winner, 'trials_finished': done_count,
                     'trials_planned': full_count,
                     'status': 'COMPLETE' if done_count == full_count else 'PROVISIONAL'})

win_cols = columns + ['trials_finished', 'trials_planned', 'status']
export('best_by_val_mse_so_far.csv', selected, win_cols)
final = [r for r in selected if r['status'] == 'COMPLETE']
export('best_by_val_mse_complete.csv', final, win_cols)

# One row per fixed lookback + horizon, with each variant's selected metrics.
wide_fields = ['dataset', 'lookback', 'horizon']
for variant in ['CycleNetMix', 'PairMix', 'MixLowRank']:
    wide_fields += [f'{variant}_val_mse', f'{variant}_test_mse',
                    f'{variant}_test_mae', f'{variant}_status']
lookup = {(r['lookback'], r['horizon'], r['variant']): r for r in selected}
wide = []
for L, H in sorted({(int(x['lookback']), int(x['horizon'])) for x in planned}):
    item = {'dataset': planned[0]['dataset'], 'lookback': L, 'horizon': H}
    for variant in ['CycleNetMix', 'PairMix', 'MixLowRank']:
        v = lookup.get((L, H, variant))
        if v:
            item.update({f'{variant}_val_mse': v['val_mse'],
                         f'{variant}_test_mse': v['test_mse'],
                         f'{variant}_test_mae': v['test_mae'],
                         f'{variant}_status': v['status']})
    wide.append(item)
export('test_comparison_by_fixed_lookback.csv', wide, wide_fields)

print(f'SUMMARY: {len(rows)}/{len(planned)} trials complete; '
      f'{len(final)}/{len(expected)} fixed-lookback groups finalized', flush=True)
if len(rows) == len(planned):
    print('ALL TRIALS COMPLETE. Final winners:')
    for x in final:
        print(f"  {x['variant']:<13} L={x['lookback']:<3} H={x['horizon']:<3} "
              f"val={x['val_mse']:.6f} test MSE={x['test_mse']:.6f} "
              f"test MAE={x['test_mae']:.6f}")
PY_SUMMARY
}

write_manifest
TOTAL=$(( $(wc -l < "$MANIFEST") - 1 ))
echo "Dataset: $DATASET | seed: $SEED | fixed lookbacks: ${LOOKBACKS[*]}"
echo "Horizons: ${HORIZONS[*]} | learning rates: ${LEARNING_RATES[*]} | batches: ${BATCH_SIZES[*]}"
echo "Models: CycleNetMix, PairMix, MixLowRank | planned trials: $TOTAL"
echo "Results: $RESULTS_DIR"

if [[ "$MODE" == '--plan' ]]; then
  echo 'Search manifest written. No training launched.'
  exit 0
fi

summarize '' yes
if [[ "$MODE" == '--summarize-only' ]]; then
  exit 0
fi

cd "$PROJECT_DIR"
if [[ ! -f ./run.py ]]; then
  echo "ERROR: run.py missing from $PROJECT_DIR" >&2
  exit 1
fi
if [[ ! -f "./dataset/$DATA_FILE" ]]; then
  echo "ERROR: ./dataset/$DATA_FILE missing" >&2
  exit 1
fi
if [[ -f ./venv/bin/activate ]]; then
  source ./venv/bin/activate
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null || { echo 'ERROR: Python not available' >&2; exit 1; }

# PairMix requires a PRECOMPUTED training-data dependency matrix.
# Use the same matrix across lookbacks only if it is intentionally applicable.
# Override e.g. PAIR_MATRIX=/path/to/your/dependency.npy sbatch script.sh
PAIR_MATRIX="${PAIR_MATRIX:-}"
if [[ -z "$PAIR_MATRIX" ]]; then
  for candidate in "./dependency_scores/Solar_dependency_L96.npy" "./dependency_scores/Solar_dependency.npy" "./dependency_scores/solar_dependency.npy"; do
    if [[ -f "$candidate" ]]; then
      PAIR_MATRIX="$candidate"
      break
    fi
  done
fi
if [[ -z "$PAIR_MATRIX" || ! -f "$PAIR_MATRIX" ]]; then
  echo 'ERROR: PairMix dependency matrix not found.' >&2
  echo 'Set PAIR_MATRIX to the existing .npy training-data dependency matrix.' >&2
  exit 1
fi
# Catch accidental dimensional mismatch, but do NOT construct scores from val/test data.
"$PYTHON_BIN" - "$PAIR_MATRIX" "$ENC_IN" <<'PY_MATRIX'
import numpy as np
import sys
array = np.load(sys.argv[1], allow_pickle=False)
expected = int(sys.argv[2])
if array.shape != (expected, expected):
    raise SystemExit(f'Invalid PairMix matrix {array.shape}; expected {(expected, expected)}')
if not np.isfinite(array).all():
    raise SystemExit('PairMix matrix contains NaN/Inf')
print('PairMix dependency matrix verified:', sys.argv[1], array.shape)
PY_MATRIX

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'WARNING: running outside SLURM. For full grid submit with: sbatch script.sh' >&2
fi

TRIAL_NUM=0
run_one() {
  local variant="$1" L="$2" H="$3" bs="$4" lr="$5" layers="$6" ratio="$7" rank="$8"
  local extra id model
  local -a extra_flags=()
  case "$variant" in
    CycleNetMix)
      model='CycleNetMixTune'
      extra="layers${layers}"
      extra_flags=(--mix_hidden "$MIX_HIDDEN" --mix_layers "$layers" --mix_dropout "$MIX_DROPOUT")
      ;;
    PairMix)
      model='CycleNetPairMix'
      extra="mr${ratio//./p}"
      extra_flags=(--mix_ratio "$ratio" --dependency_score_path "$PAIR_MATRIX" --mix_lag_window "$MIX_LAG_WINDOW" --mix_dropout "$MIX_DROPOUT")
      ;;
    MixLowRank)
      model='CycleNetMixLowRank'
      extra="rank${rank}_layers${layers}"
      extra_flags=(--mix_rank "$rank" --mix_layers "$layers" --mix_hidden "$LOWRANK_HIDDEN" --mix_dropout "$MIX_DROPOUT")
      ;;
    *) echo "Unknown variant: $variant" >&2; exit 2;;
  esac
  id="$(make_id "$variant" "$L" "$H" "$bs" "$lr" "$extra")"
  ((TRIAL_NUM += 1))
  if [[ -f "$RESULTS_DIR/records/$id.json" ]]; then
    echo "[$TRIAL_NUM/$TOTAL] SKIP completed $id"
    return
  fi
  echo "[$TRIAL_NUM/$TOTAL] TRAIN $id"
  local log="$RESULTS_DIR/logs/$id.log"
  local checkpoints="$RESULTS_DIR/checkpoints/$id"
  mkdir -p "$checkpoints"
  "$PYTHON_BIN" -u run.py \
    --is_training 1 \
    --root_path ./dataset/ --data_path "$DATA_FILE" \
    --model_id "$id" --model "$model" --data "$DATA_LOADER" \
    --features M --seq_len "$L" --pred_len "$H" \
    --enc_in "$ENC_IN" --cycle "$CYCLE" --model_type mlp --use_revin "$REVIN" \
    --train_epochs "$TRAIN_EPOCHS" --patience "$PATIENCE" --itr 1 \
    --batch_size "$bs" --learning_rate "$lr" --lradj type3 \
    --num_workers 4 --random_seed "$SEED" \
    --checkpoints "$checkpoints/" \
    "${extra_flags[@]}" \
    2>&1 | tee "$log"
  # Must have final test metrics before marking it complete.
  summarize "$id" no
}

for L in "${LOOKBACKS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for layers in "${MIX_LAYERS[@]}"; do
          run_one CycleNetMix "$L" "$H" "$bs" "$lr" "$layers" '' ''
        done
        for ratio in "${MIX_RATIOS[@]}"; do
          run_one PairMix "$L" "$H" "$bs" "$lr" '' "$ratio" ''
        done
        for rank in "${MIX_RANKS[@]}"; do
          for layers in "${MIX_LAYERS[@]}"; do
            run_one MixLowRank "$L" "$H" "$bs" "$lr" "$layers" '' "$rank"
          done
        done
      done
    done
  done
done

echo "COMPLETE: all $TOTAL experiments for $DATASET."
echo "Selected configurations: $RESULTS_DIR/best_by_val_mse_complete.csv"
echo "Test metrics by fixed lookback: $RESULTS_DIR/test_comparison_by_fixed_lookback.csv"
