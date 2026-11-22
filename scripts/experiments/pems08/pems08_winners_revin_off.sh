#!/usr/bin/env bash
#SBATCH --job-name=p08_no_revin
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=p08_no_revin_%j.out
#SBATCH --error=p08_no_revin_%j.err

# ABLATION: repeat precisely the 48 validation-MSE-selected PEMS08 mixing
# configurations from the completed RevIN-ON grid, changing ONLY use_revin=0.
# Sources: original grid's best_by_val_mse_complete.csv (not its console log).
# Idempotent/resumable: rerunning with sbatch skips valid completed records.
# Usage:
#   bash scripts/experiments/pems08/pems08_winners_revin_off.sh --plan
#   sbatch scripts/experiments/pems08/pems08_winners_revin_off.sh
#   bash scripts/experiments/pems08/pems08_winners_revin_off.sh --summarize-only
set -Eeuo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/alalade/CycleNet}"
OLD_DIR="${OLD_DIR:-${PROJECT_DIR}/fixed_lookback_grid/pems08_mix_H12_24_48_96}"
NEW_DIR="${NEW_DIR:-${PROJECT_DIR}/fixed_lookback_grid/pems08_winners_revin_off}"
MODE="${1:---run}"
case "$MODE" in --run|--plan|--summarize-only) ;; *) echo 'Usage: script.sh [--run|--plan|--summarize-only]' >&2; exit 2;; esac
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -f run.py && -f dataset/PEMS08.npz ]] || { echo 'ERROR: Missing run.py or dataset/PEMS08.npz' >&2; exit 1; }
[[ "$NEW_DIR" != "$OLD_DIR" ]] || { echo 'ERROR: NEW_DIR must differ from OLD_DIR' >&2; exit 1; }
mkdir -p "$NEW_DIR/logs" "$NEW_DIR/records" "$NEW_DIR/checkpoints"
OLD_WINNERS="$OLD_DIR/best_by_val_mse_complete.csv"
[[ -f "$OLD_WINNERS" ]] || { echo "ERROR: Missing original winners CSV: $OLD_WINNERS" >&2; exit 1; }
MANIFEST="$NEW_DIR/winners_revin_off_manifest.csv"

# Export immutable winner manifest, preserving original winning hyperparameters.
# If an existing manifest differs, refuse to mix incompatible experiments.
"$PYTHON_BIN" - "$OLD_WINNERS" "$MANIFEST" <<'PY_BUILD'
import csv, pathlib, sys, io
source, destination = map(pathlib.Path, sys.argv[1:])
with source.open(newline='') as f: rows = list(csv.DictReader(f))
expected = {(m,L,H) for m in ('CycleNetMix','PairMix','MixLowRank')
            for L in (96,192,336,720) for H in (12,24,48,96)}
keys = [(r['variant'],int(r['lookback']),int(r['horizon'])) for r in rows]
if len(rows)!=48 or set(keys)!=expected or len(set(keys))!=48:
    raise SystemExit('ERROR: expected exactly 48 distinct original winning configurations')
if any(r.get('status')!='COMPLETE' for r in rows):
    raise SystemExit('ERROR: some original winners are not marked COMPLETE')
for r in rows:
    if r.get('trials_finished')!=r.get('trials_planned'):
        raise SystemExit('ERROR: incomplete original tuning grid for '+r['experiment_id'])
    if r.get('seed')!='2024': raise SystemExit('ERROR: unexpected seed: '+r['experiment_id'])
    for key in ('learning_rate','batch_size','val_mse','test_mse','test_mae'):
        if not r.get(key): raise SystemExit('ERROR: missing '+key+' in '+r['experiment_id'])
    r['original_experiment_id']=r.pop('experiment_id')
    r['experiment_id']=r['original_experiment_id']+'_revin0'
    r['use_revin']='0'
rows.sort(key=lambda r: (int(r['lookback']),int(r['horizon']),r['variant']))
fields = ['experiment_id','original_experiment_id','variant','lookback','horizon',
          'batch_size','learning_rate','mix_layers','mix_ratio','mix_rank',
          'mix_hidden','mix_dropout','seed','use_revin',
          'val_mse','test_mse','test_mae']
out = io.StringIO()
writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore')
writer.writeheader();writer.writerows(rows)
content=out.getvalue()
# Compare the actual configuration, not fluctuating reference metrics.
# It is safe to refresh reference test/validation numbers only when every
# hyperparameter and source experiment ID is identical.
CONFIG_FIELDS = [
    'experiment_id','original_experiment_id','variant','lookback','horizon',
    'batch_size','learning_rate','mix_layers','mix_ratio','mix_rank',
    'mix_hidden','mix_dropout','seed','use_revin'
]
if destination.exists():
    with destination.open(newline='') as f:
        prior = list(csv.DictReader(f))
    def keyed(records):
        return {(r['variant'], int(r['lookback']), int(r['horizon'])): r for r in records}
    a, b = keyed(prior), keyed(rows)
    changes = []
    if len(prior) != len(rows) or set(a) != set(b):
        changes.append(f'Experiment groups differ: old={len(prior)}, new={len(rows)}')
    for group in sorted(set(a) & set(b)):
        for field in CONFIG_FIELDS:
            if a[group].get(field, '') != b[group].get(field, ''):
                changes.append(
                    f'{group}: {field}: existing={a[group].get(field)!r}, '
                    f'new={b[group].get(field)!r}'
                )
    if changes:
        print('ERROR: Existing RevIN-OFF runs were created from DIFFERENT winning configurations.', file=sys.stderr)
        print('The script will not mix those runs with the current winners.', file=sys.stderr)
        for c in changes[:12]:
            print('  - '+c, file=sys.stderr)
        if len(changes)>12:
            print(f'  ... plus {len(changes)-12} additional differences', file=sys.stderr)
        print('Keep the old output folder. To launch a clean version, use:', file=sys.stderr)
        new_dir = str(destination.parent) + '_v2'
        print('  sbatch --export=ALL,NEW_DIR='+new_dir+' scripts/experiments/pems08/pems08_winners_revin_off.sh', file=sys.stderr)
        raise SystemExit(1)
    if destination.read_text() != content:
        print('INFO: Winner configurations unchanged; refreshing original validation/test reference metrics.')
# This never overwrites an ablation run: only the metadata in the CSV manifest
# is updated after matching all model-defining hyperparameters.
destination.write_text(content)
print('Validated original winners:', len(rows),'(all 48 groups COMPLETE)')
print('Ablation manifest:', destination)
PY_BUILD

# --plan and --summarize-only do not launch training. Both print a concise summary.
summarize() {
  "$PYTHON_BIN" - "$MANIFEST" "$NEW_DIR" <<'PY_SUMMARIZE'
import csv, json, pathlib, sys, re, math, os
manifest, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
with manifest.open(newline='') as f: rows=list(csv.DictReader(f))
num = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
def extract(path):
    s=path.read_text(errors='replace')
    f=s.rsplit('FINAL TEST RESULTS',1)
    if len(f)!=2: raise ValueError('no FINAL TEST RESULTS')
    a=re.search(rf'(?im)^\s*MSE\s*:\s*({num})\s*$',f[-1]); b=re.search(rf'(?im)^\s*MAE\s*:\s*({num})\s*$',f[-1])
    epochs=[(int(e),float(v)) for e,v in re.findall(rf'(?m)^\s*Epoch:\s*(\d+),\s*Steps:.*?Val\s+MSE:\s*({num})',s)]
    if not (a and b and epochs): raise ValueError('missing test metrics or validation epochs')
    ep, val=min(epochs,key=lambda z:(z[1],z[0]))
    result={'best_epoch':ep,'val_mse':val,'test_mse':float(a.group(1)),'test_mae':float(b.group(1))}
    if not all(map(math.isfinite,result.values())): raise ValueError('non-finite metric')
    return result
completed=[]
for row in rows:
    rid=row['experiment_id']; record=root/'records'/f'{rid}.json'; log=root/'logs'/f'{rid}.log'
    if not record.exists() and log.exists():
        try: metrics=extract(log)
        except ValueError: pass
        else:
            tmp=record.with_suffix('.json.tmp');tmp.write_text(json.dumps(metrics,indent=2)+'\n');os.replace(tmp,record)
    if record.exists():
        metrics=json.loads(record.read_text())
        on_mse=float(row['test_mse']);on_mae=float(row['test_mae'])
        new={**row,**metrics,
             'revin_on_test_mse':on_mse,'revin_on_test_mae':on_mae,
             'mse_improvement_vs_revin_on_pct':(on_mse-metrics['test_mse'])/on_mse*100,
             'mae_improvement_vs_revin_on_pct':(on_mae-metrics['test_mae'])/on_mae*100}
        # Restore naming clarity: val_mse from selected RevIN-ON config is kept separately.
        new['revin_on_val_mse']=row['val_mse']
        completed.append(new)
completed.sort(key=lambda r:(int(r['lookback']),int(r['horizon']),r['variant']))
fieldnames=['variant','lookback','horizon','batch_size','learning_rate','mix_layers','mix_ratio','mix_rank','seed',
            'original_experiment_id','experiment_id','revin_on_val_mse','revin_on_test_mse','revin_on_test_mae',
            'best_epoch','val_mse','test_mse','test_mae','mse_improvement_vs_revin_on_pct','mae_improvement_vs_revin_on_pct']
dest=root/'revin_off_vs_on_comparison.csv';temp=dest.with_suffix('.csv.tmp')
with temp.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore');writer.writeheader();writer.writerows(completed)
os.replace(temp,dest)
print(f'REVIN-OFF COMPLETED: {len(completed)}/48')
print('Results:',dest)
for x in completed:
    print(f"  {x['variant']:<11} L={x['lookback']:>3} H={x['horizon']:>2} "
          f"OFF MSE={x['test_mse']:.7f} MAE={x['test_mae']:.7f} "
          f"vs ON MSE={x['mse_improvement_vs_revin_on_pct']:+.2f}% "
          f"MAE={x['mae_improvement_vs_revin_on_pct']:+.2f}%")
PY_SUMMARIZE
}

if [[ "$MODE" == --plan ]]; then
  echo 'Planned 48 models in manifest (same L/H, LR, batch and mixer settings; use_revin=0):'
  "$PYTHON_BIN" - "$MANIFEST" <<'PY_PLAN'
import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
 print(f"{r['variant']:<11} L={r['lookback']:>3} H={r['horizon']:>2} "
       f"bs={r['batch_size']:>2} lr={r['learning_rate']:<7} "
       f"layers={r['mix_layers'] or '-':<2} ratio={r['mix_ratio'] or '-':<4} rank={r['mix_rank'] or '-':<2}")
PY_PLAN
  exit 0
fi
if [[ "$MODE" == --summarize-only ]]; then summarize; exit 0; fi

PAIR_MATRIX="${PAIR_MATRIX:-$PROJECT_DIR/dependency_scores/PEMS08_dependency.npy}"
[[ -f "$PAIR_MATRIX" ]] || { echo "ERROR: dependency matrix missing: $PAIR_MATRIX" >&2; exit 1; }
"$PYTHON_BIN" - "$PAIR_MATRIX" <<'PY_CHECK'
import numpy as np,sys
x=np.load(sys.argv[1],allow_pickle=False)
if x.shape!=(170,170) or not np.isfinite(x).all():
 raise SystemExit('ERROR: PEMS08 dependency matrix must be finite 170x170')
print('PairMix dependency matrix:',sys.argv[1],x.shape)
PY_CHECK

TOTAL=48; COUNT=0
while IFS=',' read -r id old_id variant L H bs lr layers ratio rank hidden dropout seed revin old_val old_mse old_mae; do
  [[ "$id" == experiment_id ]] && continue
  ((COUNT+=1))
  RECORD="$NEW_DIR/records/$id.json"; LOG="$NEW_DIR/logs/$id.log"
  if [[ -f "$RECORD" ]]; then echo "[$COUNT/$TOTAL] SKIP $id"; continue; fi
  echo "[$COUNT/$TOTAL] TRAIN $id (original winner $old_id; RevIN 1 -> 0)"
  mkdir -p "$NEW_DIR/checkpoints/$id"
  extra=()
  case "$variant" in
    CycleNetMix)
      model=CycleNetMixTune
      extra=(--mix_hidden "$hidden" --mix_layers "$layers" --mix_dropout "$dropout") ;;
    PairMix)
      model=CycleNetPairMix
      extra=(--mix_ratio "$ratio" --dependency_score_path "$PAIR_MATRIX" --mix_lag_window 4 --mix_dropout "$dropout") ;;
    MixLowRank)
      model=CycleNetMixLowRank
      extra=(--mix_rank "$rank" --mix_layers "$layers" --mix_hidden "$hidden" --mix_dropout "$dropout") ;;
    *) echo "ERROR: unknown model $variant" >&2; exit 1;;
  esac
  "$PYTHON_BIN" -u run.py \
    --is_training 1 \
    --root_path ./dataset/ --data_path PEMS08.npz \
    --model_id "$id" --model "$model" --data PEMS \
    --features M --seq_len "$L" --pred_len "$H" \
    --enc_in 170 --cycle 288 --model_type mlp --use_revin 0 \
    --train_epochs 30 --patience 5 --itr 1 \
    --batch_size "$bs" --learning_rate "$lr" --lradj type3 \
    --num_workers 4 --random_seed "$seed" \
    --checkpoints "$NEW_DIR/checkpoints/$id/" \
    "${extra[@]}" 2>&1 | tee "$LOG"
  # Fail closed rather than accidentally treating a truncated run as complete.
  "$PYTHON_BIN" - "$LOG" "$RECORD" <<'PY_RECORD'
import re,json,math,sys,pathlib,os
log, record=map(pathlib.Path,sys.argv[1:]);s=log.read_text(errors='replace')
parts=s.rsplit('FINAL TEST RESULTS',1)
if len(parts)!=2: raise SystemExit('ERROR: training finished without FINAL TEST RESULTS')
num=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
mse=re.search(rf'(?im)^\s*MSE\s*:\s*({num})\s*$',parts[-1])
mae=re.search(rf'(?im)^\s*MAE\s*:\s*({num})\s*$',parts[-1])
val=[(int(e),float(v)) for e,v in re.findall(rf'(?m)^\s*Epoch:\s*(\d+),\s*Steps:.*?Val\s+MSE:\s*({num})',s)]
if not(mse and mae and val): raise SystemExit('ERROR: missing validation/test metrics')
epoch, v=min(val,key=lambda z:(z[1],z[0]))
d={'best_epoch':epoch,'val_mse':v,'test_mse':float(mse.group(1)),'test_mae':float(mae.group(1))}
if not all(math.isfinite(x) for x in d.values()): raise SystemExit('ERROR: nonfinite metrics')
tmp=record.with_suffix('.json.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,record)
print(f"RECORDED: val={v:.7f}, test MSE={d['test_mse']:.7f}, MAE={d['test_mae']:.7f}")
PY_RECORD
  summarize
done < "$MANIFEST"
summarize
echo "FINISHED: $COUNT/48 runs checked. Results: $NEW_DIR/revin_off_vs_on_comparison.csv"
