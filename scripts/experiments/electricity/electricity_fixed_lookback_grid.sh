#!/bin/bash
#SBATCH --job-name=elec_grid
#SBATCH --partition=STUD
#SBATCH --array=0-47%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=elec_grid_%A_%a.out
#SBATCH --error=elec_grid_%A_%a.err

set -euo pipefail
cd "${PROJECT_DIR:-/home/alalade/CycleNet}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/venv/bin/python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$PYTHON_BIN" - "${1:-run}" <<'PY'
import csv
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path.cwd()
OUT = ROOT / 'fixed_lookback_grid' / 'electricity'
DEP = ROOT / 'dependency_scores' / 'Electricity_dependency.npy'
DATA = ROOT / 'dataset' / 'electricity.csv'
LOOKBACKS = HORIZONS = (96, 192, 336, 720)
MODELS = ('CycleNetMix', 'PairMix', 'MixLowRank')
BATCHES = (32, 64)
LRS = ('0.0001', '0.0005', '0.001', '0.002')
LAYERS = (1, 2)
RATIOS = ('0.2', '0.4', '0.6')
RANKS = (16, 32, 64)
SEED = 2024
GROUPS = list(itertools.product(LOOKBACKS, HORIZONS, MODELS))
FIELDS = ['dataset','model','lookback','horizon','batch_size','learning_rate',
          'mix_layers','mix_ratio','mix_rank','mix_hidden','mix_dropout','seed',
          'run_id','validation_mse','test_mse','test_mae','best_epoch','log']


def trials(L, H, model):
    for bs, lr in itertools.product(BATCHES, LRS):
        stem = f'Electricity_{model}_L{L}_H{H}_bs{bs}_lr{lr.replace(".", "p")}'
        settings = ([(l,None,None) for l in LAYERS] if model == 'CycleNetMix'
                    else [(1,r,None) for r in RATIOS] if model == 'PairMix'
                    else [(l,None,r) for r,l in itertools.product(RANKS, LAYERS)])
        for layers, ratio, rank in settings:
            suffix = (f'layers{layers}' if model == 'CycleNetMix' else
                      f'mr{ratio.replace(".", "p")}' if model == 'PairMix' else
                      f'rank{rank}_layers{layers}')
            yield dict(dataset='Electricity',model=model,lookback=L,horizon=H,
                       batch_size=bs,learning_rate=lr,mix_layers=layers,
                       mix_ratio=ratio,mix_rank=rank,
                       mix_hidden=64 if model=='CycleNetMix' else 128,
                       mix_dropout=0.1,seed=SEED,
                       run_id=f'{stem}_{suffix}_seed{SEED}')


def result_path(t):
    return OUT/'results'/f'{t["run_id"]}.json'


def load_result(t):
    try:
        r=json.loads(result_path(t).read_text())
        if any(r.get(k)!=v for k,v in t.items()): return None
        if not all(math.isfinite(float(r[k])) and float(r[k])>=0 for k in
                   ('validation_mse','test_mse','test_mae')): return None
        return r
    except (OSError, ValueError, KeyError, TypeError):
        return None


def atomic_json(path, r):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(r,indent=2)+'\n')
    os.replace(tmp,path)


def csv_write(path, rows, fields):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    os.replace(tmp,path)


def summary():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        winners=[]; all_results=[]; coverage=[]; provisional=[]
        for L,H,model in GROUPS:
            grid=list(trials(L,H,model))
            done=[r for t in grid if (r:=load_result(t)) is not None]
            all_results.extend(done)
            complete=len(done)==len(grid)
            coverage.append(dict(model=model,lookback=L,horizon=H,
                                 completed=len(done),expected=len(grid),
                                 status='COMPLETE' if complete else 'INCOMPLETE'))
            if done:
                # deterministic tie-break using configuration ID, never test metrics
                best=min(done,key=lambda r:(r['validation_mse'],r['run_id']))
                provisional.append(dict(best,completed=len(done),expected=len(grid),
                                       status='COMPLETE' if complete else 'PROVISIONAL'))
                if complete: winners.append(best)
        csv_write(OUT/'all_trials.csv',all_results,FIELDS)
        csv_write(OUT/'winning_configurations.csv',winners,FIELDS)
        csv_write(OUT/'provisional_winners.csv',provisional,FIELDS+['completed','expected','status'])
        csv_write(OUT/'coverage.csv',coverage,['model','lookback','horizon','completed','expected','status'])
        print(f'SUMMARY: {len(all_results)}/1408 trials complete; {len(winners)}/48 groups finalized',flush=True)
        return len(all_results),len(winners)


def parse_log(path):
    s=path.read_text(errors='replace')
    # Use the exact checkpoint validation score and final test metrics printed
    # by the local CycleNet experiment implementation. No test-based selection.
    v=re.findall(r'BEST VALIDATION CHECKPOINT\s*=+\s*Best epoch:\s*(\d+)\s*\n\s*Validation MSE:\s*([0-9.eE+\-]+)',s)
    m=re.findall(r'FINAL TEST RESULTS\s*=+\s*MSE:\s*([0-9.eE+\-]+)\s*\n\s*MAE:\s*([0-9.eE+\-]+)',s)
    if len(v)!=1 or len(m)!=1: raise ValueError('Expected exactly one validation checkpoint and one final test block')
    epoch=int(v[0][0]); vals=tuple(map(float,(v[0][1],m[0][0],m[0][1])))
    if not all(math.isfinite(x) and x>=0 for x in vals): raise ValueError('Invalid metric')
    return dict(best_epoch=epoch,validation_mse=vals[0],test_mse=vals[1],test_mae=vals[2])


def command(t):
    model=t['model']; model_arg={'CycleNetMix':'CycleNetMixTune','PairMix':'CycleNetPairMix','MixLowRank':'CycleNetMixLowRank'}[model]
    ckpt=OUT/'checkpoints'/t['run_id']
    ckpt.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u','run.py','--is_training','1','--root_path','./dataset/',
         '--data_path','electricity.csv','--model_id',t['run_id'],'--model',model_arg,
         '--data','custom','--features','M','--seq_len',str(t['lookback']),
         '--label_len','0','--pred_len',str(t['horizon']),'--enc_in','321','--cycle','168',
         '--model_type','mlp','--d_model','512','--use_revin','1','--train_epochs','30',
         '--patience','5','--itr','1','--batch_size',str(t['batch_size']),
         '--learning_rate',t['learning_rate'],'--lradj','type3','--num_workers','4',
         '--random_seed',str(SEED),'--dropout','0','--mix_hidden',str(t['mix_hidden']),
         '--mix_dropout',str(t['mix_dropout']),'--mix_layers',str(t['mix_layers']),
         '--checkpoints',str(ckpt)+'/']
    if model=='PairMix': cmd+=['--mix_ratio',t['mix_ratio'],'--dependency_score_path',str(DEP)]
    if model=='MixLowRank': cmd+=['--mix_rank',str(t['mix_rank'])]
    return cmd

mode=sys.argv[1]
if mode=='--plan':
    for i,(L,H,model) in enumerate(GROUPS): print(i,model,L,H,len(list(trials(L,H,model))))
    print('TOTAL:',sum(len(list(trials(*g))) for g in GROUPS))
    sys.exit(0)
if mode=='--summary':
    summary(); sys.exit(0)
if mode!='run': raise SystemExit('Usage: bash scripts/experiments/electricity/electricity_fixed_lookback_grid.sh [--plan|--summary]')

# Before submitting the long grid, validate the dataset and dependency matrix.
import numpy as np
import pandas as pd
assert (ROOT/'run.py').is_file(), 'run.py missing'
assert DATA.is_file(), f'Missing {DATA}'
header=pd.read_csv(DATA,nrows=0).columns
assert 'date' in header and 'OT' in header, 'Expected date and OT columns for the custom data loader'
assert len(header)-1==321, f'Expected 321 data channels, found {len(header)-1}'
D=np.load(DEP,allow_pickle=False)
assert D.shape==(321,321) and np.isfinite(D).all(), 'Invalid dependency matrix'
assert np.allclose(D,D.T,atol=1e-6) and np.allclose(np.diag(D),0,atol=1e-6), 'Invalid matrix symmetry/diagonal'
for d in ('results','logs','checkpoints','locks'):(OUT/d).mkdir(parents=True,exist_ok=True)
idx=int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'))
if not 0<=idx<len(GROUPS):raise SystemExit('Submit with sbatch (array task 0-47), or use --plan / --summary.')
L,H,model=GROUPS[idx]; grid=list(trials(L,H,model))
print(f'Python: {sys.executable}\nElectricity | L={L} H={H} | {model} | {len(grid)} trials\nDependency: {DEP}')
failures=0
# Prevent accidental duplicate submissions from running the same group concurrently.
with (OUT/'locks'/f'group_{idx}.lock').open('a') as group_lock:
    try: fcntl.flock(group_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit(f'Group {idx} is already running; refusing duplicate execution')
    for n,t in enumerate(grid,1):
        rid=t['run_id']
        if (r:=load_result(t)) is not None:
            print(f'[{n}/{len(grid)}] SKIP completed {rid}')
            continue
        log=OUT/'logs'/f'{rid}.log'
        if log.exists():
            # Keep previous incomplete/failed attempt for auditing.
            import time
            log.rename(log.with_suffix(f'.attempt_{time.time_ns()}.log'))
        print(f'[{n}/{len(grid)}] TRAIN {rid} | {log}',flush=True)
        try:
            with log.open('w') as f:
                p=subprocess.run(command(t),stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            if p.returncode!=0: raise RuntimeError(f'run.py exited {p.returncode}')
            r=dict(t,**parse_log(log),log=str(log))
            atomic_json(result_path(t),r)
            print(f'VERIFIED: {rid} | Val MSE={r["validation_mse"]:.8f} | Test MSE={r["test_mse"]:.8f} | Test MAE={r["test_mae"]:.8f}',flush=True)
        except (OSError,RuntimeError,ValueError) as e:
            failures+=1; print(f'FAILED: {rid}: {e}; see {log}',flush=True)
        summary()
    done=[r for t in grid if (r:=load_result(t)) is not None]
    if len(done)==len(grid):
        best=min(done,key=lambda r:(r['validation_mse'],r['run_id']))
        print('FINAL WINNER (LOWEST VALIDATION MSE):\n'+json.dumps(best,indent=2))
    else: print(f'INCOMPLETE GROUP: {len(done)}/{len(grid)}; resubmit this array task to retry missing trials')
    summary()
    if failures or len(done)!=len(grid): sys.exit(1)
PY
