#!/usr/bin/env bash
#SBATCH --job-name=srp_etth1_mix
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=srp_etth1_mix_%j.out
#SBATCH --error=srp_etth1_mix_%j.err

# ETT ETTh1: independent, fixed-lookback grid search on two CycleNet extensions.
# CycleNetMixTune: mix_layers=1,2; hidden width=64; dropout=0.1
# CycleNetMixLowRank: mix_rank=2,4,6; mix_layers=1,2; dropout=0.1
# For each model x L x H, choose the lowest training-time validation MSE,
# then report that selected trial's TEST MSE and TEST MAE (never select by test).
# 1024 trials = 4 lookbacks x 4 horizons x 2 batch sizes x 4 LRs x (2+6).
# Re-submit the identical script to resume; validated completed trials are skipped.
# Usage: bash script.sh --plan | --summarize-only ; sbatch script.sh
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/alalade/CycleNet}"
DATASET="ETTh1"
DATA_LOADER="ETTh1"
DATA_FILE="ETTh1.csv"
ENC_IN=7
CYCLE=24
SEED="${SEED:-2024}"
REVIN="${REVIN:-1}"
D_MODEL=512
MIX_HIDDEN=64
LOWRANK_HIDDEN=128   # compatible with run.py; LowRank uses --mix_rank
MIX_DROPOUT=0.1
TRAIN_EPOCHS=30
PATIENCE=5
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/fixed_lookback_grid/etth1_mix_only}"
MODE="${1:---run}"

read -r -a LOOKBACKS      <<< "${LOOKBACKS_OVERRIDE:-96 192 336 720}"
read -r -a HORIZONS       <<< "${HORIZONS_OVERRIDE:-96 192 336 720}"
read -r -a BATCH_SIZES    <<< "${BATCH_SIZES_OVERRIDE:-128 256}"
read -r -a LEARNING_RATES <<< "${LEARNING_RATES_OVERRIDE:-0.0001 0.0005 0.001 0.002}"
read -r -a MIX_LAYERS     <<< "${MIX_LAYERS_OVERRIDE:-1 2}"
# 7 input channels: actual low-rank bottlenecks, unlike ranks 16/32/64.
# To match the older ETTm1 grid instead, set MIX_RANKS_OVERRIDE='16 32 64'.
read -r -a MIX_RANKS      <<< "${MIX_RANKS_OVERRIDE:-2 4 6}"

if [[ "$MODE" != "--run" && "$MODE" != "--plan" && "$MODE" != "--summarize-only" ]]; then
  echo "Usage: bash $0 [--run | --plan | --summarize-only]" >&2; exit 2
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || true)"
  fi
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  echo "ERROR: No working Python. Set PYTHON_BIN=/home/alalade/CycleNet/venv/bin/python" >&2; exit 1;
}

mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/records" "$RESULTS_DIR/checkpoints"
MANIFEST="$RESULTS_DIR/manifest.csv"

make_id() {
  local v="$1" L="$2" H="$3" bs="$4" lr="$5" extra="$6"
  printf '%s_%s_L%s_H%s_bs%s_lr%s_%s_revin%s_seed%s' \
    "$DATASET" "$v" "$L" "$H" "$bs" "${lr//./p}" "$extra" "$REVIN" "$SEED"
}

write_manifest() {
  local temp="${MANIFEST}.tmp.$$" L H bs lr layers rank id
  printf '%s\n' 'experiment_id,dataset,variant,lookback,horizon,learning_rate,batch_size,mix_layers,mix_rank,mix_hidden,mix_dropout,cycle,d_model,use_revin,seed,train_epochs,patience' > "$temp"
  for L in "${LOOKBACKS[@]}"; do
    for H in "${HORIZONS[@]}"; do
      for bs in "${BATCH_SIZES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
          for layers in "${MIX_LAYERS[@]}"; do
            id="$(make_id CycleNetMix "$L" "$H" "$bs" "$lr" "layers${layers}")"
            printf '%s,%s,CycleNetMix,%s,%s,%s,%s,%s,,%s,%s,%s,%s,%s,%s,%s,%s\n' \
              "$id" "$DATASET" "$L" "$H" "$lr" "$bs" "$layers" "$MIX_HIDDEN" "$MIX_DROPOUT" "$CYCLE" "$D_MODEL" "$REVIN" "$SEED" "$TRAIN_EPOCHS" "$PATIENCE" >> "$temp"
          done
          for rank in "${MIX_RANKS[@]}"; do
            for layers in "${MIX_LAYERS[@]}"; do
              id="$(make_id MixLowRank "$L" "$H" "$bs" "$lr" "rank${rank}_layers${layers}")"
              printf '%s,%s,MixLowRank,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$id" "$DATASET" "$L" "$H" "$lr" "$bs" "$layers" "$rank" "$LOWRANK_HIDDEN" "$MIX_DROPOUT" "$CYCLE" "$D_MODEL" "$REVIN" "$SEED" "$TRAIN_EPOCHS" "$PATIENCE" >> "$temp"
            done
          done
        done
      done
    done
  done
  if [[ -f "$MANIFEST" ]] && ! cmp -s "$temp" "$MANIFEST"; then
    echo 'ERROR: Existing manifest differs from requested experiment settings.' >&2
    echo 'Do not mix old/new grids. Set RESULTS_DIR to a new directory.' >&2
    diff -u "$MANIFEST" "$temp" | head -n 35 >&2 || true
    rm -f "$temp"
    exit 1
  fi
  mv "$temp" "$MANIFEST"
}

summarize() {
  local new_id="${1:-}" recover="${2:-no}"
  "$PYTHON_BIN" - "$RESULTS_DIR" "$new_id" "$recover" <<'PY_SUMMARY'
from pathlib import Path
from collections import defaultdict
import csv, json, math, os, re, sys
root=Path(sys.argv[1]); new_id=sys.argv[2]; recover=sys.argv[3]=='yes'
with (root/'manifest.csv').open(newline='') as fp:
    planned=list(csv.DictReader(fp))
ids={x['experiment_id']:x for x in planned}
if len(ids)!=len(planned): raise ValueError('Duplicate IDs in manifest')
NUM=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

def parse_trial(run_id):
    raw=(root/'logs'/f'{run_id}.log').read_text(errors='replace')
    if 'FINAL TEST RESULTS' not in raw:
        raise ValueError('Missing FINAL TEST RESULTS marker')
    end=raw.rsplit('FINAL TEST RESULTS',1)[-1]
    mse=re.search(rf'(?im)^\s*MSE\s*:\s*({NUM})\s*$',end)
    mae=re.search(rf'(?im)^\s*MAE\s*:\s*({NUM})\s*$',end)
    if not mse or not mae: raise ValueError('Missing final test MSE/MAE')
    matches=[(int(e),float(m)) for e,m in re.findall(
        rf'(?m)^\s*Epoch:\s*(\d+),\s*Steps:.*?Val\s+MSE:\s*({NUM})',raw)]
    if not matches: raise ValueError('Missing epoch validation MSE')
    epoch,val=min(matches,key=lambda z:(z[1],z[0]))
    # If the runner prints an explicit training checkpoint summary, cross-check it.
    lines=[x for x in raw.splitlines() if x.startswith('VALIDATION RESULT |')]
    if lines:
        m=re.search(rf'best_val_mse=({NUM})',lines[-1])
        if m and abs(float(m.group(1))-val)>2e-6:
            raise ValueError('Inconsistent validation metric in log')
    ans={'best_epoch':epoch,'val_mse':val,'test_mse':float(mse.group(1)),'test_mae':float(mae.group(1))}
    if not all(map(math.isfinite,ans.values())): raise ValueError('Nonfinite metrics')
    return ans

def record(run_id,res):
    p=root/'records'/f'{run_id}.json'
    tmp=p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(res,indent=2)+'\n')
    os.replace(tmp,p)

if recover:
    for run_id in ids:
        p=root/'records'/f'{run_id}.json'
        if not p.exists() and (root/'logs'/f'{run_id}.log').exists():
            try: record(run_id,parse_trial(run_id));print('RECOVERED',run_id,flush=True)
            except (ValueError,OSError) as e: print('INCOMPLETE',run_id,str(e),flush=True)
if new_id:
    if new_id not in ids: raise ValueError('Trial missing from manifest')
    result=parse_trial(new_id)
    record(new_id,result)
    print(f'VERIFIED {new_id} | val={result["val_mse"]:.8f} | test MSE={result["test_mse"]:.8f} | MAE={result["test_mae"]:.8f}',flush=True)

fields=list(planned[0])+['best_epoch','val_mse','test_mse','test_mae']
rows=[]
for item in planned:
    p=root/'records'/f'{item["experiment_id"]}.json'
    if p.exists():
        metrics=json.loads(p.read_text())
        rows.append({**item,**metrics})

def export(file,items,columns):
    dest=root/file; tmp=dest.with_suffix(dest.suffix+'.tmp')
    with tmp.open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=columns,extrasaction='ignore')
        w.writeheader();w.writerows(items)
    os.replace(tmp,dest)
export('all_completed_trials.csv',rows,fields)
expected=defaultdict(set); groups=defaultdict(list)
for x in planned: expected[(x['variant'],int(x['lookback']),int(x['horizon']))].add(x['experiment_id'])
for x in rows: groups[(x['variant'],int(x['lookback']),int(x['horizon']))].append(x)
winners=[]
for group in sorted(groups,key=lambda k:(k[1],k[2],k[0])):
    candidates=groups[group]
    best=min(candidates,key=lambda x:(float(x['val_mse']),x['experiment_id']))
    done=len(candidates); total=len(expected[group])
    winners.append({**best,'trials_finished':done,'trials_planned':total,
                    'status':'COMPLETE' if done==total else 'PROVISIONAL'})
winfields=fields+['trials_finished','trials_planned','status']
export('best_by_val_mse_so_far.csv',winners,winfields)
final=[x for x in winners if x['status']=='COMPLETE']
export('best_by_val_mse_complete.csv',final,winfields)
lookup={(int(x['lookback']),int(x['horizon']),x['variant']):x for x in winners}
widefields=['dataset','lookback','horizon']
for variant in ('CycleNetMix','MixLowRank'):
    widefields += [f'{variant}_val_mse',f'{variant}_test_mse',f'{variant}_test_mae',f'{variant}_status']
wide=[]
for L,H in sorted({(int(x['lookback']),int(x['horizon'])) for x in planned}):
    entry={'dataset':planned[0]['dataset'],'lookback':L,'horizon':H}
    for variant in ('CycleNetMix','MixLowRank'):
        w=lookup.get((L,H,variant))
        if w: entry.update({f'{variant}_val_mse':w['val_mse'],
                            f'{variant}_test_mse':w['test_mse'],
                            f'{variant}_test_mae':w['test_mae'],
                            f'{variant}_status':w['status']})
    wide.append(entry)
export('test_comparison_by_fixed_lookback.csv',wide,widefields)
print(f'SUMMARY: {len(rows)}/{len(planned)} trials complete; {len(final)}/{len(expected)} groups finalized',flush=True)
if len(rows)==len(planned):
    print('ALL TRIALS COMPLETE — winning configurations, selected by validation MSE:',flush=True)
    for w in final:
        print(f'  {w["variant"]:12s} L={w["lookback"]:>3} H={w["horizon"]:>3} Val={float(w["val_mse"]):.6f} Test MSE={w["test_mse"]:.6f} MAE={w["test_mae"]:.6f}',flush=True)
PY_SUMMARY
}

write_manifest
TOTAL=$(( $(wc -l < "$MANIFEST") - 1 ))
echo "Dataset=$DATASET | cycle=$CYCLE | channels=$ENC_IN | RevIN=$REVIN | seed=$SEED"
echo "Lookbacks: ${LOOKBACKS[*]} | Horizons: ${HORIZONS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]} | LRs: ${LEARNING_RATES[*]} | Mix layers: ${MIX_LAYERS[*]} | ranks: ${MIX_RANKS[*]}"
echo "Models: CycleNetMix, MixLowRank | Planned trials: $TOTAL | Output: $RESULTS_DIR"
if [[ "$MODE" == '--plan' ]]; then
    echo "PLAN ONLY. Preview of trial IDs:"; head -n 9 "$MANIFEST";exit 0
fi
summarize '' yes
if [[ "$MODE" == '--summarize-only' ]]; then exit 0;fi

cd "$PROJECT_DIR" || { echo "Project directory not found: $PROJECT_DIR" >&2;exit 1; }
[[ -f ./run.py ]] || { echo 'ERROR: run.py not found in project directory' >&2;exit 1; }
[[ -f "./dataset/$DATA_FILE" ]] || {
    echo "ERROR: ./dataset/$DATA_FILE missing; download the appropriate ETT dataset first." >&2;exit 1;
}
# Protect against accidentally invoking unsupported model/dataset identifiers.
"$PYTHON_BIN" - <<'PY_PRE'
from pathlib import Path
assert Path('models/CycleNetMixTune.py').exists(), 'Missing models/CycleNetMixTune.py'
assert Path('models/CycleNetMixLowRank.py').exists(), 'Missing models/CycleNetMixLowRank.py'
PY_PRE
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'WARNING: Running outside SLURM; submit with sbatch for full grid.' >&2
fi

TRIAL_NUM=0
run_one() {
    local variant="$1" L="$2" H="$3" bs="$4" lr="$5" layers="$6" rank="$7" extra id model
    local -a flags=()
    case "$variant" in
      CycleNetMix)
        model=CycleNetMixTune;extra="layers${layers}"
        flags=(--mix_hidden "$MIX_HIDDEN" --mix_layers "$layers" --mix_dropout "$MIX_DROPOUT")
        ;;
      MixLowRank)
        model=CycleNetMixLowRank;extra="rank${rank}_layers${layers}"
        flags=(--mix_hidden "$LOWRANK_HIDDEN" --mix_rank "$rank" --mix_layers "$layers" --mix_dropout "$MIX_DROPOUT")
        ;;
      *) echo "Unknown variant $variant" >&2;exit 2;;
    esac
    id="$(make_id "$variant" "$L" "$H" "$bs" "$lr" "$extra")"
    ((TRIAL_NUM+=1))
    local log="$RESULTS_DIR/logs/$id.log" ckp="$RESULTS_DIR/checkpoints/$id"
    if [[ -f "$RESULTS_DIR/records/$id.json" ]]; then
      echo "[$TRIAL_NUM/$TOTAL] SKIP completed $id";return 0
    fi
    echo "[$TRIAL_NUM/$TOTAL] TRAIN $id"
    mkdir -p "$ckp"
    "$PYTHON_BIN" -u run.py \
        --is_training 1 \
        --model_id "$id" --model "$model" --data "$DATA_LOADER" \
        --root_path ./dataset/ --data_path "$DATA_FILE" --features M \
        --seq_len "$L" --pred_len "$H" --enc_in "$ENC_IN" \
        --cycle "$CYCLE" --model_type mlp --d_model "$D_MODEL" \
        --use_revin "$REVIN" --random_seed "$SEED" \
        --train_epochs "$TRAIN_EPOCHS" --patience "$PATIENCE" \
        --batch_size "$bs" --learning_rate "$lr" --lradj type3 \
        --itr 1 --num_workers 4 --checkpoints "$ckp/" \
        "${flags[@]}" 2>&1 | tee "$log"
    summarize "$id" no
}
for L in "${LOOKBACKS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for layers in "${MIX_LAYERS[@]}"; do
          run_one CycleNetMix "$L" "$H" "$bs" "$lr" "$layers" ''
        done
        for rank in "${MIX_RANKS[@]}"; do
          for layers in "${MIX_LAYERS[@]}"; do
            run_one MixLowRank "$L" "$H" "$bs" "$lr" "$layers" "$rank"
          done
        done
      done
    done
  done
done

echo "FINISHED: $DATASET all $TOTAL trials."
echo "Best configurations: $RESULTS_DIR/best_by_val_mse_complete.csv"
echo "Comparison: $RESULTS_DIR/test_comparison_by_fixed_lookback.csv"
