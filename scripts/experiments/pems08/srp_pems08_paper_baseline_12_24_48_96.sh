#!/usr/bin/env bash
#SBATCH --job-name=p08_paper_base
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=p08_paper_base_%j.out
#SBATCH --error=p08_paper_base_%j.err
set -Eeuo pipefail
# Official CycleNet PEMS08 MLP settings, held fixed while varying L and H.
# Source: https://github.com/ACAT-SCUT/CycleNet/blob/main/scripts/CycleNet/PEMS/pems08.sh
# Official settings originally reported at L=96. L=192/336/720 are our extension.
PROJECT_DIR="${PROJECT_DIR:-/home/alalade/CycleNet}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/fixed_lookback_grid/pems08_paper_baseline_H12_24_48_96}"
MODE="${1:---run}"
LOOKBACKS=(96 192 336 720)
HORIZONS=(12 24 48 96)
SEED=2024
LR=0.005
BS=64
CYCLE=288
REVIN=0
EPOCHS=30
PATIENCE=5
mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/checkpoints" "$RESULTS_DIR/records"
cd "$PROJECT_DIR"
if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then PYTHON_BIN="$PROJECT_DIR/venv/bin/python"; else PYTHON_BIN="${PYTHON_BIN:-python3}"; fi
"$PYTHON_BIN" --version
if [[ "$MODE" == '--run' && ( ! -f run.py || ! -f dataset/PEMS08.npz ) ]]; then echo 'Missing run.py or dataset/PEMS08.npz' >&2; exit 1; fi
summarize() {
 "$PYTHON_BIN" - "$RESULTS_DIR" <<'PY'
import csv, math, re, json, os, sys
from pathlib import Path
root=Path(sys.argv[1]); results=[]
number=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
for L in (96,192,336,720):
 for H in (12,24,48,96):
  id=f'PEMS08_CycleNet_paper_L{L}_H{H}_seed2024'
  log=root/'logs'/f'{id}.log'; record=root/'records'/f'{id}.json'
  if not record.exists() and log.exists():
   s=log.read_text(errors='replace')
   if 'FINAL TEST RESULTS' in s:
    tail=s.rsplit('FINAL TEST RESULTS',1)[1]
    mse=re.search(rf'(?m)^\s*MSE:\s*({number})\s*$',tail)
    mae=re.search(rf'(?m)^\s*MAE:\s*({number})\s*$',tail)
    vals=re.findall(rf'(?m)^\s*Epoch:\s*(\d+),\s*Steps:.*?Val\s+MSE:\s*({number})',s)
    if mse and mae and vals:
     epoch,vmse=min(((int(e),float(v)) for e,v in vals),key=lambda x:(x[1],x[0]))
     out=dict(best_epoch=epoch,val_mse=vmse,test_mse=float(mse.group(1)),test_mae=float(mae.group(1)))
     if all(math.isfinite(v) for v in (vmse,out['test_mse'],out['test_mae'])):
      tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');os.replace(tmp,record)
      print('VERIFIED:',id, out, flush=True)
  if record.exists():
   data=json.loads(record.read_text())
   results.append(dict(experiment_id=id,dataset='PEMS08',variant='CycleNet',lookback=L,horizon=H,
      learning_rate=0.005,batch_size=64,cycle=288,use_revin=0,seed=2024,**data,
      trials_finished=1,trials_planned=1,status='COMPLETE'))
fields=['experiment_id','dataset','variant','lookback','horizon','learning_rate','batch_size','cycle','use_revin','seed','best_epoch','val_mse','test_mse','test_mae','trials_finished','trials_planned','status']
for name in ['best_by_val_mse_so_far.csv','best_by_val_mse_complete.csv']:
 with (root/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
print(f'CycleNet original-config baseline: {len(results)}/16 complete',flush=True)
PY
}
summarize
case "$MODE" in
 --plan) echo 'CycleNet, 4 L x 4 H = 16 runs, one fixed paper configuration per pair';exit 0;;
 --summarize-only) exit 0;;
 --run) ;;
 *) echo "Invalid mode: $MODE" >&2;exit 2;;
esac
for L in "${LOOKBACKS[@]}"; do
 for H in "${HORIZONS[@]}"; do
  id="PEMS08_CycleNet_paper_L${L}_H${H}_seed2024"
  if [[ -f "$RESULTS_DIR/records/$id.json" ]];then echo "SKIP $id";continue;fi
  echo "TRAIN $id"
  mkdir -p "$RESULTS_DIR/checkpoints/$id"
  "$PYTHON_BIN" -u run.py \
   --is_training 1 --root_path ./dataset/ --data_path PEMS08.npz \
   --model_id "$id" --model CycleNet --data PEMS --features M \
   --seq_len "$L" --pred_len "$H" --enc_in 170 --cycle "$CYCLE" \
   --model_type mlp --use_revin "$REVIN" \
   --train_epochs "$EPOCHS" --patience "$PATIENCE" --itr 1 \
   --batch_size "$BS" --learning_rate "$LR" --random_seed "$SEED" \
   --num_workers 4 --checkpoints "$RESULTS_DIR/checkpoints/$id/" \
   2>&1 | tee "$RESULTS_DIR/logs/$id.log"
  summarize
  if [[ ! -f "$RESULTS_DIR/records/$id.json" ]];then echo "ERROR: Missing completed metrics for $id" >&2;exit 1;fi
 done
done
echo "Finished all 16 fixed-paper CycleNet PEMS08 experiments."
