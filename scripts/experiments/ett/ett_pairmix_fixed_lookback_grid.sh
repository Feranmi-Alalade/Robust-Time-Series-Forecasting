#!/usr/bin/env bash
#SBATCH --job-name=ett_pairmix
#SBATCH --partition=STUD
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=ett_pairmix_%A_%a.out
#SBATCH --error=ett_pairmix_%A_%a.err

# Fixed-lookback CycleNetPairMix grid for ETTm2, ETTh1, and ETTh2.
# One array task per dataset: 0=ETTm2; 1=ETTh1; 2=ETTh2.
# Each (dataset, lookback L, horizon H) chooses the lowest training-time
# validation MSE among the 24 parameter configurations. Report TEST MSE
# and MAE only for that same validation-selected configuration.
#
# Default per-dataset: 4 L x 4 H x 2 batches x 4 learning rates x 3 mix ratios
# = 384 trials; three datasets = 1,152 trials.
#
# Usage:
#   SLURM_ARRAY_TASK_ID=0 bash scripts/experiments/ett/ett_pairmix_fixed_lookback_grid.sh --plan
#   sbatch scripts/experiments/ett/ett_pairmix_fixed_lookback_grid.sh
#   SLURM_ARRAY_TASK_ID=0 bash scripts/experiments/ett/ett_pairmix_fixed_lookback_grid.sh --summarize-only
# Re-submit the SAME script with sbatch to resume unfinished trials.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/alalade/CycleNet}"
MODE="${1:---run}"
case "$MODE" in
  --run|--plan|--summarize-only) ;;
  *) echo "Usage: bash $0 [--run | --plan | --summarize-only]" >&2; exit 2 ;;
esac

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0) DATASET=ETTm2; DATA_FILE=ETTm2.csv; CYCLE=96 ;;
  1) DATASET=ETTh1; DATA_FILE=ETTh1.csv; CYCLE=24 ;;
  2) DATASET=ETTh2; DATA_FILE=ETTh2.csv; CYCLE=24 ;;
  *) echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID; expected 0, 1, or 2" >&2; exit 2 ;;
esac

DATA_LOADER="$DATASET"
ENC_IN=7
SEED="${SEED:-2024}"
REVIN="${REVIN:-1}"
D_MODEL=512
MIX_DROPOUT="${MIX_DROPOUT:-0.1}"
MIX_LAG_WINDOW=4
TRAIN_EPOCHS=30
PATIENCE=5

read -r -a LOOKBACKS      <<< "${LOOKBACKS_OVERRIDE:-96 192 336 720}"
read -r -a HORIZONS       <<< "${HORIZONS_OVERRIDE:-96 192 336 720}"
read -r -a BATCH_SIZES    <<< "${BATCH_SIZES_OVERRIDE:-128 256}"
read -r -a LEARNING_RATES <<< "${LEARNING_RATES_OVERRIDE:-0.0001 0.0005 0.001 0.002}"
read -r -a MIX_RATIOS     <<< "${MIX_RATIOS_OVERRIDE:-0.2 0.4 0.6}"

MATRIX="${MATRIX_OVERRIDE:-${PROJECT_DIR}/dependency_scores/${DATASET}_dependency.npy}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/fixed_lookback_grid/${DATASET,,}_pairmix_only}"
MANIFEST="$RESULTS_DIR/manifest.csv"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || true)"
  fi
fi

make_id() {
  local L="$1" H="$2" bs="$3" lr="$4" ratio="$5"
  printf '%s_PairMix_L%s_H%s_bs%s_lr%s_mr%s_revin%s_seed%s' \
    "$DATASET" "$L" "$H" "$bs" "${lr//./p}" "${ratio//./p}" "$REVIN" "$SEED"
}

TOTAL=$(( ${#LOOKBACKS[@]} * ${#HORIZONS[@]} * ${#BATCH_SIZES[@]} * ${#LEARNING_RATES[@]} * ${#MIX_RATIOS[@]} ))
NUM_GROUPS=$(( ${#LOOKBACKS[@]} * ${#HORIZONS[@]} ))
PER_GROUP=$(( ${#BATCH_SIZES[@]} * ${#LEARNING_RATES[@]} * ${#MIX_RATIOS[@]} ))

printf '\n===== %s CycleNetPairMix fixed-lookback grid =====\n' "$DATASET"
echo "Dataset=$DATASET | data=$DATA_FILE | channels=$ENC_IN | cycle=$CYCLE | RevIN=$REVIN | seed=$SEED"
echo "Lookbacks: ${LOOKBACKS[*]} | Horizons: ${HORIZONS[*]}"
echo "Batches: ${BATCH_SIZES[*]} | Learning rates: ${LEARNING_RATES[*]} | Mix ratios: ${MIX_RATIOS[*]}"
echo "Groups=$NUM_GROUPS | trials/group=$PER_GROUP | total trials=$TOTAL"
echo "Dependency matrix: $MATRIX"
echo "Result directory: $RESULTS_DIR"

if [[ "$MODE" == --plan ]]; then
  echo 'PLAN ONLY — no files changed and no training launched.'
  echo "First trial: $(make_id "${LOOKBACKS[0]}" "${HORIZONS[0]}" "${BATCH_SIZES[0]}" "${LEARNING_RATES[0]}" "${MIX_RATIOS[0]}")"
  exit 0
fi

[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  echo "ERROR: Python not found; set PYTHON_BIN=/home/alalade/CycleNet/venv/bin/python" >&2; exit 1;
}

# Do not rebuild or mutate manifests in summarize-only mode.
if [[ "$MODE" == --summarize-only ]]; then
  [[ -f "$MANIFEST" ]] || { echo "ERROR: manifest missing: $MANIFEST" >&2; exit 1; }
else
  cd "$PROJECT_DIR" || { echo "ERROR: project directory absent: $PROJECT_DIR" >&2; exit 1; }
  [[ -f ./run.py ]] || { echo 'ERROR: run.py is missing' >&2; exit 1; }
  [[ -f "./dataset/$DATA_FILE" ]] || { echo "ERROR: dataset/$DATA_FILE is missing" >&2; exit 1; }
  [[ -f ./models/CycleNetPairMix.py ]] || { echo 'ERROR: models/CycleNetPairMix.py is missing' >&2; exit 1; }
  [[ -f "$MATRIX" ]] || {
    echo "ERROR: $MATRIX does not exist; finish scripts/experiments/ett/ett_dependency_matrices.sh first." >&2;
    exit 1;
  }

  "$PYTHON_BIN" - "$MATRIX" "$ENC_IN" <<'PY_MATRIX'
import numpy as np, sys
path, C = sys.argv[1], int(sys.argv[2])
D = np.load(path, allow_pickle=False)
assert D.shape == (C, C), f'Wrong dependency shape {D.shape}; expected {(C,C)}'
assert np.isfinite(D).all(), 'Dependency matrix contains NaN/Inf'
assert np.allclose(D, D.T, atol=1e-6), 'Dependency matrix must be symmetric'
assert np.allclose(np.diag(D), 0, atol=1e-6), 'Dependency matrix diagonal must be zero'
print('Verified existing training-residual dependency matrix:', path, '| shape:', D.shape)
PY_MATRIX

  # Include the dependency-matrix fingerprint in the manifest: if matrix data
  # change, continuing an old grid would invalidate the comparison.
  MATRIX_SHA="$(sha256sum "$MATRIX" | awk '{print $1}')"
  mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/records" "$RESULTS_DIR/checkpoints"

  write_manifest() {
    local temp="${MANIFEST}.tmp.$$"
    local L H bs lr ratio id
    echo 'experiment_id,dataset,variant,lookback,horizon,learning_rate,batch_size,mix_ratio,mix_dropout,mix_lag_window,cycle,d_model,use_revin,seed,train_epochs,patience,dependency_sha256' > "$temp"
    for L in "${LOOKBACKS[@]}"; do
      for H in "${HORIZONS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
          for lr in "${LEARNING_RATES[@]}"; do
            for ratio in "${MIX_RATIOS[@]}"; do
              id="$(make_id "$L" "$H" "$bs" "$lr" "$ratio")"
              printf '%s,%s,PairMix,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$id" "$DATASET" "$L" "$H" "$lr" "$bs" "$ratio" "$MIX_DROPOUT" \
                "$MIX_LAG_WINDOW" "$CYCLE" "$D_MODEL" "$REVIN" "$SEED" \
                "$TRAIN_EPOCHS" "$PATIENCE" "$MATRIX_SHA" >> "$temp"
            done
          done
        done
      done
    done
    if [[ -f "$MANIFEST" ]] && ! cmp -s "$temp" "$MANIFEST"; then
      echo 'ERROR: Existing manifest differs from requested grid or dependency matrix.' >&2
      echo 'Set RESULTS_DIR to a NEW directory; never mix configurations.' >&2
      diff -u "$MANIFEST" "$temp" | head -n 30 >&2 || true
      rm -f "$temp"
      exit 1
    fi
    mv "$temp" "$MANIFEST"
  }
  write_manifest
fi

# Read each COMPLETED trial's TEST MSE/MAE and minimum training-time Val MSE.
# Best-validation-checkpoint re-evaluation is NOT used to rank hyperparameters.
# Rebuild all summaries so resumptions remain consistent and recover completed
# trials whose jobs ended before the JSON completion record was written.
summarize() {
  local new_id="${1:-}" recover="${2:-no}"
  "$PYTHON_BIN" - "$RESULTS_DIR" "$new_id" "$recover" <<'PY_SUMMARY'
from pathlib import Path
from collections import defaultdict
import csv, json, math, os, re, sys

root=Path(sys.argv[1]); new_id=sys.argv[2]; recover=sys.argv[3]=='yes'
with (root/'manifest.csv').open(newline='') as f:
    planned=list(csv.DictReader(f))
if not planned: raise ValueError('Empty manifest')
ids={r['experiment_id']:r for r in planned}
if len(ids)!=len(planned): raise ValueError('Duplicate run IDs in manifest')
NUM=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

def parse_trial(run_id):
    log=root/'logs'/f'{run_id}.log'
    raw=log.read_text(errors='replace')
    if 'FINAL TEST RESULTS' not in raw:
        raise ValueError('Missing FINAL TEST RESULTS marker')
    end=raw.rsplit('FINAL TEST RESULTS',1)[-1]
    mse=re.search(rf'(?im)^\s*MSE\s*:\s*({NUM})\s*$',end)
    mae=re.search(rf'(?im)^\s*MAE\s*:\s*({NUM})\s*$',end)
    if not mse or not mae: raise ValueError('Missing final test MSE/MAE')
    vals=[(int(e),float(m)) for e,m in re.findall(
        rf'(?m)^\s*Epoch:\s*(\d+),\s*Steps:.*?Val\s+MSE:\s*({NUM})',raw)]
    if not vals: raise ValueError('Missing epoch validation MSE')
    epoch,val=min(vals,key=lambda p:(p[1],p[0]))
    lines=[x for x in raw.splitlines() if x.startswith('VALIDATION RESULT |')]
    if lines:
        m=re.search(rf'best_val_mse=({NUM})',lines[-1])
        if m and abs(float(m.group(1))-val)>2e-6:
            raise ValueError('Inconsistent training-time best-validation MSE')
    result={'best_epoch':epoch,'val_mse':val,
            'test_mse':float(mse.group(1)),'test_mae':float(mae.group(1))}
    if not all(math.isfinite(x) for x in result.values()):
        raise ValueError('Non-finite trial metric')
    return result

def save_record(run_id,metrics):
    path=root/'records'/f'{run_id}.json'
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(metrics,indent=2)+'\n')
    os.replace(tmp,path)

if recover:
    for run_id in ids:
        record=root/'records'/f'{run_id}.json'
        log=root/'logs'/f'{run_id}.log'
        if not record.exists() and log.exists():
            try:
                save_record(run_id,parse_trial(run_id))
                print('RECOVERED',run_id,flush=True)
            except (OSError,ValueError) as exc:
                print('INCOMPLETE (will rerun)',run_id,str(exc),flush=True)
if new_id:
    if new_id not in ids: raise ValueError(f'Run ID not in manifest: {new_id}')
    result=parse_trial(new_id)
    save_record(new_id,result)
    print(f'VERIFIED {new_id} | Val MSE={result["val_mse"]:.8f} | Test MSE={result["test_mse"]:.8f} | Test MAE={result["test_mae"]:.8f}',flush=True)

fields=list(planned[0])+['best_epoch','val_mse','test_mse','test_mae']
rows=[]
for meta in planned:
    record=root/'records'/f'{meta["experiment_id"]}.json'
    if record.exists():
        metrics=json.loads(record.read_text())
        rows.append({**meta,**metrics})

def export(filename,content,columns):
    dest=root/filename
    tmp=dest.with_suffix(dest.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(content)
    os.replace(tmp,dest)

export('all_completed_trials.csv',rows,fields)
expected=defaultdict(set); actual=defaultdict(list)
for meta in planned:
    key=(meta['variant'],int(meta['lookback']),int(meta['horizon']))
    expected[key].add(meta['experiment_id'])
for row in rows:
    key=(row['variant'],int(row['lookback']),int(row['horizon']))
    actual[key].append(row)

winners=[]
for key in sorted(actual,key=lambda k:(k[1],k[2],k[0])):
    choices=actual[key]
    best=min(choices,key=lambda r:(float(r['val_mse']),r['experiment_id']))
    count=len(choices); planned_count=len(expected[key])
    winners.append({**best,'trials_finished':count,'trials_planned':planned_count,
                    'status':'COMPLETE' if count==planned_count else 'PROVISIONAL'})
winfields=fields+['trials_finished','trials_planned','status']
export('best_by_val_mse_so_far.csv',winners,winfields)
complete=[x for x in winners if x['status']=='COMPLETE']
export('best_by_val_mse_complete.csv',complete,winfields)

# One compact row per fixed L and H, with all model-selection details in the
# winners CSV; this CSV carries the selected test metrics for comparisons.
lookup={(int(x['lookback']),int(x['horizon'])):x for x in winners}
wide=[]
for L,H in sorted({(int(x['lookback']),int(x['horizon'])) for x in planned}):
    found=lookup.get((L,H))
    out={'dataset':planned[0]['dataset'],'lookback':L,'horizon':H}
    if found:
        out.update({'val_mse':found['val_mse'],'test_mse':found['test_mse'],
                    'test_mae':found['test_mae'],'mix_ratio':found['mix_ratio'],
                    'learning_rate':found['learning_rate'],
                    'batch_size':found['batch_size'],'status':found['status']})
    wide.append(out)
export('test_comparison_by_fixed_lookback.csv',wide,
       ['dataset','lookback','horizon','val_mse','test_mse','test_mae',
        'mix_ratio','learning_rate','batch_size','status'])

print(f'SUMMARY: {len(rows)}/{len(planned)} trials complete; '
      f'{len(complete)}/{len(expected)} lookback-horizon winners finalized',flush=True)
if len(rows)==len(planned):
    print('ALL TRIALS COMPLETE — validation-selected winners:',flush=True)
    for w in complete:
        print(f'  {w["dataset"]} L={w["lookback"]:>3} H={w["horizon"]:>3} '
              f'ratio={w["mix_ratio"]} bs={w["batch_size"]} lr={w["learning_rate"]} '
              f'Val MSE={float(w["val_mse"]):.6f} '
              f'Test MSE={float(w["test_mse"]):.6f} MAE={float(w["test_mae"]):.6f}',flush=True)
PY_SUMMARY
}

summarize '' yes
if [[ "$MODE" == --summarize-only ]]; then
  echo "Summaries regenerated in: $RESULTS_DIR"
  exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'WARNING: Run mode invoked outside SLURM; for full grid use sbatch.' >&2
fi

cd "$PROJECT_DIR"
COUNTER=0
run_one() {
  local L="$1" H="$2" bs="$3" lr="$4" ratio="$5" id log ckpt
  id="$(make_id "$L" "$H" "$bs" "$lr" "$ratio")"
  ((COUNTER+=1))
  if [[ -f "$RESULTS_DIR/records/$id.json" ]]; then
    echo "[$COUNTER/$TOTAL] SKIP completed $id"
    return 0
  fi
  echo "[$COUNTER/$TOTAL] TRAIN $id"
  log="$RESULTS_DIR/logs/$id.log"
  ckpt="$RESULTS_DIR/checkpoints/$id"
  mkdir -p "$ckpt"
  "$PYTHON_BIN" -u run.py \
    --is_training 1 \
    --model_id "$id" --model CycleNetPairMix --data "$DATA_LOADER" \
    --root_path ./dataset/ --data_path "$DATA_FILE" --features M \
    --seq_len "$L" --label_len 0 --pred_len "$H" --enc_in "$ENC_IN" \
    --cycle "$CYCLE" --model_type mlp --d_model "$D_MODEL" \
    --use_revin "$REVIN" --random_seed "$SEED" \
    --train_epochs "$TRAIN_EPOCHS" --patience "$PATIENCE" \
    --batch_size "$bs" --learning_rate "$lr" --lradj type3 \
    --itr 1 --num_workers 4 --checkpoints "$ckpt/" \
    --dependency_score_path "$MATRIX" --mix_ratio "$ratio" \
    --mix_dropout "$MIX_DROPOUT" --mix_lag_window "$MIX_LAG_WINDOW" \
    2>&1 | tee "$log"
  summarize "$id" no
}

for L in "${LOOKBACKS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for ratio in "${MIX_RATIOS[@]}"; do
          run_one "$L" "$H" "$bs" "$lr" "$ratio"
        done
      done
    done
  done
done

echo "FINISHED: $DATASET all $TOTAL trials."
echo "Selected hyperparameters: $RESULTS_DIR/best_by_val_mse_complete.csv"
echo "Test summary by L/H: $RESULTS_DIR/test_comparison_by_fixed_lookback.csv"
