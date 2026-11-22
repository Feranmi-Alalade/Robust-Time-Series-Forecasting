#!/usr/bin/env bash
#SBATCH --job-name=cy_L192
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-6%2
#SBATCH --output=cyclenet_L192_%A_%a.out
#SBATCH --error=cyclenet_L192_%A_%a.err

# CycleNet/MLP baseline at a FIXED lookback of 192 for seven datasets.
# Four forecasts per dataset; 28 experiments in total (seed 2024).
# One SLURM array task = one dataset. Completed runs are skipped.
# Important: CycleNet's paper did not publish an L=192 setting.
# We deliberately reuse the authors' L=336 model/optimizer settings
# (the authors' PEMS08 settings for PEMS08), changing seq_len ONLY.
# See README in the corresponding CycleNet official scripts.
#
# Usage:
#   bash scripts/experiments/baseline/cyclenet_L192_all_datasets.sh --plan
#   sbatch scripts/experiments/baseline/cyclenet_L192_all_datasets.sh
#   squeue -u "$USER" -r
#   bash scripts/experiments/baseline/cyclenet_L192_all_datasets.sh --summarize-only
#
# Outputs:
#   fixed_lookback_grid/cyclenet_L192_baselines/<dataset>/logs/*.log
#   fixed_lookback_grid/cyclenet_L192_baselines/<dataset>/records/*.json
#   fixed_lookback_grid/cyclenet_L192_baselines/<dataset>/results.csv
#   fixed_lookback_grid/cyclenet_L192_baselines/cyclenet_L192_all_results.csv

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/alalade/CycleNet}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/fixed_lookback_grid/cyclenet_L192_baselines}"
SEED="${SEED:-2024}"
LOOKBACK=192
MODE="${1:---run}"

DATASETS=(ETTm1 Solar Weather PEMS08 ETTm2 ETTh1 ETTh2)

usage() {
  echo "Usage: bash $0 [--plan | --summarize-only | --run]"
  echo "Submit all datasets: sbatch $0"
}

# Sets the dataset-specific authors' configurations.
configure_dataset() {
  local dataset="$1"
  DATASET="$dataset"
  case "$dataset" in
    ETTm1)
      DATA_LOADER=ETTm1; DATA_FILE=ETTm1.csv; ENC_IN=7; CYCLE=96
      BATCH=256; LR=0.002; REVIN=1; HORIZONS=(96 192 336 720) ;;
    Solar)
      DATA_LOADER=Solar; DATA_FILE=solar_AL.txt; ENC_IN=137; CYCLE=144
      BATCH=64; LR=0.002; REVIN=0; HORIZONS=(96 192 336 720) ;;
    Weather)
      DATA_LOADER=custom; DATA_FILE=weather.csv; ENC_IN=21; CYCLE=144
      BATCH=256; LR=0.002; REVIN=1; HORIZONS=(96 192 336 720) ;;
    PEMS08)
      DATA_LOADER=PEMS; DATA_FILE=PEMS08.npz; ENC_IN=170; CYCLE=288
      BATCH=64; LR=0.005; REVIN=0; HORIZONS=(12 24 48 96) ;;
    ETTm2)
      DATA_LOADER=ETTm2; DATA_FILE=ETTm2.csv; ENC_IN=7; CYCLE=96
      BATCH=256; LR=0.002; REVIN=1; HORIZONS=(96 192 336 720) ;;
    ETTh1)
      DATA_LOADER=ETTh1; DATA_FILE=ETTh1.csv; ENC_IN=7; CYCLE=24
      BATCH=256; LR=0.002; REVIN=1; HORIZONS=(96 192 336 720) ;;
    ETTh2)
      DATA_LOADER=ETTh2; DATA_FILE=ETTh2.csv; ENC_IN=7; CYCLE=24
      BATCH=256; LR=0.002; REVIN=1; HORIZONS=(96 192 336 720) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
  esac
}

slug_lr() { printf '%s' "${1//./p}"; }
make_id() {
  local H="$1"
  printf '%s_CycleNet_L%s_H%s_bs%s_lr%s_revin%s_seed%s' \
    "$DATASET" "$LOOKBACK" "$H" "$BATCH" "$(slug_lr "$LR")" "$REVIN" "$SEED"
}

if [[ "$MODE" == '--plan' ]]; then
  echo "CycleNet MLP | fixed L=192 | seed=$SEED | 7 datasets x 4 horizons = 28 trials"
  echo "Note: for L192, using authors' L336 hyperparameters (PEMS: authors' PEMS08 settings)."
  for i in "${!DATASETS[@]}"; do
    configure_dataset "${DATASETS[$i]}"
    printf 'Array %d | %-7s | H=%-14s cycle=%-3s channels=%-3s bs=%-3s lr=%-6s RevIN=%s\n' \
      "$i" "$DATASET" "${HORIZONS[*]}" "$CYCLE" "$ENC_IN" "$BATCH" "$LR" "$REVIN"
  done
  exit 0
fi

[[ "$MODE" == '--run' || "$MODE" == '--summarize-only' ]] || { usage >&2; exit 2; }

# Summarize JSON records already collected. This also works after the array finishes.
summarize_records() {
  local scope="${1:-all}"
  python - "$RESULTS_ROOT" "$scope" <<'PY_SUMMARIZE'
import csv, json, os, sys
from pathlib import Path
root = Path(sys.argv[1]); scope = sys.argv[2]
datasets = ['ETTm1','Solar','Weather','PEMS08','ETTm2','ETTh1','ETTh2']
fields = ['dataset','model','lookback','horizon','cycle','enc_in','batch_size',
          'learning_rate','use_revin','seed','best_epoch','val_mse','test_mse','test_mae',
          'source_config','run_id']

def atomic_csv(destination, rows):
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp = destination.with_name(destination.name + '.tmp')
    with tmp.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp,destination)

all_rows=[]
for ds in datasets:
    if scope != 'all' and ds != scope:
        continue
    rec_dir=root / ds / 'records'
    rows=[]
    if rec_dir.exists():
        for f in rec_dir.glob('*.json'):
            x=json.loads(f.read_text())
            rows.append({k:x.get(k,'') for k in fields})
    rows.sort(key=lambda r:(int(r['lookback']),int(r['horizon']),int(r['seed'])))
    atomic_csv(root / ds / 'results.csv',rows)
    print(f'{ds}: {len(rows)}/4 completed for seed 2024' if scope != 'all' else
          f'{ds}: {len(rows)} completed record(s)')
    all_rows.extend(rows)
if scope == 'all':
    order={name:i for i,name in enumerate(datasets)}
    all_rows.sort(key=lambda r:(order[r['dataset']],int(r['lookback']),int(r['horizon']),int(r['seed'])))
    atomic_csv(root / 'cyclenet_L192_all_results.csv', all_rows)
    print('Combined CSV:',root / 'cyclenet_L192_all_results.csv')
PY_SUMMARIZE
}

if [[ "$MODE" == '--summarize-only' ]]; then
  if [[ ! -d "$PROJECT_DIR/venv" ]]; then
    echo "ERROR: Python venv not found at $PROJECT_DIR/venv" >&2; exit 1
  fi
  # shellcheck source=/dev/null
  source "$PROJECT_DIR/venv/bin/activate"
  summarize_records all
  exit 0
fi

# Training mode: require an array task to avoid accidentally launching 28 GPU jobs on login.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo 'ERROR: Training must be submitted with sbatch as an array job.' >&2
  echo "Use: sbatch $0" >&2
  exit 2
fi
if ! [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-6]$ ]]; then
  echo "ERROR: Unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

configure_dataset "${DATASETS[$SLURM_ARRAY_TASK_ID]}"
cd "$PROJECT_DIR"
[[ -f run.py ]] || { echo "ERROR: $PROJECT_DIR/run.py missing" >&2; exit 1; }
[[ -f "dataset/$DATA_FILE" ]] || { echo "ERROR: Missing dataset/$DATA_FILE" >&2; exit 1; }
[[ -f venv/bin/activate ]] || { echo 'ERROR: venv/bin/activate missing' >&2; exit 1; }
# shellcheck source=/dev/null
source venv/bin/activate
command -v python >/dev/null || { echo 'ERROR: python not found after activating venv' >&2; exit 1; }

DATA_DIR="$RESULTS_ROOT/$DATASET"
mkdir -p "$DATA_DIR/logs" "$DATA_DIR/records" "$DATA_DIR/checkpoints"

echo "============================================================="
echo "Dataset=$DATASET | model=CycleNet MLP | L=$LOOKBACK | seed=$SEED"
echo "Horizons=${HORIZONS[*]} | cycle=$CYCLE | channels=$ENC_IN"
echo "batch=$BATCH | lr=$LR | RevIN=$REVIN | array_id=$SLURM_ARRAY_TASK_ID"
echo "Python=$(command -v python)"
echo "Results: $DATA_DIR"
echo "============================================================="

# A record is written only after validation and final TEST metrics were parsed.
capture_trial() {
  local id="$1" H="$2" log="$3" record="$4"
  python - "$id" "$DATASET" "$H" "$LOOKBACK" "$ENC_IN" "$CYCLE" \
      "$BATCH" "$LR" "$REVIN" "$SEED" "$log" "$record" <<'PY_CAPTURE'
import json, math, os, re, sys
from pathlib import Path
(id,dataset,h,lookback,channels,cycle,batch,lr,revin,seed,log_path,record_path) = sys.argv[1:]
s=Path(log_path).read_text(errors='replace')
number=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
if 'FINAL TEST RESULTS' in s:
    tail=s.rsplit('FINAL TEST RESULTS',1)[1]
    m=re.search(rf'(?im)^\s*MSE\s*:\s*({number})\s*$',tail)
    a=re.search(rf'(?im)^\s*MAE\s*:\s*({number})\s*$',tail)
    if not (m and a):
        raise SystemExit('ERROR: Incomplete FINAL TEST RESULTS block')
    tmse,tmae=float(m.group(1)),float(a.group(1))
else:
    # Older CycleNet run.py writes 'mse:..., mae:...' instead.
    matches=re.findall(rf'(?im)^\s*mse\s*:\s*({number})\s*,\s*mae\s*:\s*({number})',s)
    if not matches:
        raise SystemExit('ERROR: Missing final TEST MSE/MAE in log')
    tmse,tmae=map(float,matches[-1])
vals = [(int(e),float(v)) for e,v in re.findall(
    rf'(?m)^\s*Epoch:\s*(\d+),\s*Steps:.*?Val\s+MSE:\s*({number})',s)]
if not vals:
    vals = [(int(e),float(v)) for e,v in re.findall(
        rf'(?m)^\s*Epoch:\s*(\d+),\s*Steps:.*?Vali\s+Loss:\s*({number})',s)]
if not vals:
    raise SystemExit('ERROR: No epoch-level validation losses found')
epoch,best_val=min(vals,key=lambda x:(x[1],x[0]))
if not all(map(math.isfinite,(tmse,tmae,best_val))):
    raise SystemExit('ERROR: Non-finite metric encountered')
row=dict(run_id=id,dataset=dataset,model='CycleNet',lookback=int(lookback),
         horizon=int(h),enc_in=int(channels),cycle=int(cycle),batch_size=int(batch),
         learning_rate=float(lr),use_revin=int(revin),seed=int(seed),
         best_epoch=epoch,val_mse=best_val,test_mse=tmse,test_mae=tmae,
         source_config='CycleNet authors L336 parameters applied at L192' if dataset!='PEMS08'
                       else 'CycleNet authors PEMS08 L96 parameters applied at L192')
dest=Path(record_path);dest.parent.mkdir(parents=True,exist_ok=True)
tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n')
os.replace(tmp,dest)
print(f'VERIFIED {id}: best Val MSE={best_val:.7f} (epoch {epoch}), '
      f'Test MSE={tmse:.7f}, Test MAE={tmae:.7f}',flush=True)
PY_CAPTURE
}

for H in "${HORIZONS[@]}"; do
  RUN_ID="$(make_id "$H")"
  RECORD="$DATA_DIR/records/$RUN_ID.json"
  LOG="$DATA_DIR/logs/$RUN_ID.log"
  CHECKPOINT="$DATA_DIR/checkpoints/$RUN_ID"

  if [[ -f "$RECORD" ]]; then
    echo "SKIP completed $RUN_ID"
    continue
  fi

  # Recover an interrupted post-processing step if training had already finished.
  if [[ -s "$LOG" ]] && capture_trial "$RUN_ID" "$H" "$LOG" "$RECORD" >/dev/null 2>&1; then
    echo "RECOVERED completed training from log: $RUN_ID"
    summarize_records "$DATASET"
    continue
  fi

  echo "============================================================="
  echo "TRAIN: $RUN_ID"
  echo "============================================================="
  mkdir -p "$CHECKPOINT"

  # pipefail ensures that Python errors propagate through tee.
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path "$DATA_FILE" \
    --model_id "$RUN_ID" \
    --model CycleNet \
    --data "$DATA_LOADER" \
    --features M \
    --seq_len "$LOOKBACK" \
    --label_len 0 \
    --pred_len "$H" \
    --enc_in "$ENC_IN" \
    --cycle "$CYCLE" \
    --model_type mlp \
    --d_model 512 \
    --use_revin "$REVIN" \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 \
    --batch_size "$BATCH" \
    --learning_rate "$LR" \
    --random_seed "$SEED" \
    --num_workers 4 \
    --checkpoints "$CHECKPOINT/" \
    2>&1 | tee "$LOG"

  capture_trial "$RUN_ID" "$H" "$LOG" "$RECORD"
  summarize_records "$DATASET"
done

echo "DONE: $DATASET, fixed lookback $LOOKBACK."
echo "Per-dataset results: $DATA_DIR/results.csv"
echo 'After ALL array tasks finish, combine with:'
echo "bash $0 --summarize-only"
