import torch
import types
from models import CycleNetMix, CycleNetMixLag, CycleNetMixLowRank, CycleNetTrend, CycleNetTrendMix

# ---------------------------------------------------------------
# Dataset settings (must match what was actually used to train)
# ---------------------------------------------------------------
DATASETS = [
    {"name": "traffic",  "data": "custom", "enc_in": 862, "cycle": 168, "seq_len": 720},
    {"name": "weather",  "data": "custom", "enc_in": 21,  "cycle": 144, "seq_len": 336},
    {"name": "ETTm1",    "data": "ETTm1",  "enc_in": 7,   "cycle": 96,  "seq_len": 336},
]

# ---------------------------------------------------------------
# Model settings: which module to use, which gates it has, and any
# extra config attributes that model's __init__ reads via getattr.
# ---------------------------------------------------------------
MODELS = [
    {"name": "CycleNetMix",        "module": CycleNetMix,        "has_trend": False, "has_mix": True,
     "extra": {"mix_hidden": 128, "mix_layers": 1, "mix_dropout": 0.1}},
    {"name": "CycleNetMixLag",     "module": CycleNetMixLag,     "has_trend": False, "has_mix": True,
     "extra": {"mix_hidden": 128, "mix_layers": 1, "mix_dropout": 0.1, "mix_lag_window": 4}},
    {"name": "CycleNetMixLowRank", "module": CycleNetMixLowRank, "has_trend": False, "has_mix": True,
     "extra": {"mix_rank": 16, "mix_layers": 1, "mix_dropout": 0.1}},
    {"name": "CycleNetTrend",      "module": CycleNetTrend,      "has_trend": True,  "has_mix": False,
     "extra": {"moving_avg": 25}},
    {"name": "CycleNetTrendMix",   "module": CycleNetTrendMix,   "has_trend": True,  "has_mix": True,
     "extra": {"mix_rank": 16, "mix_layers": 1, "mix_dropout": 0.1, "moving_avg": 25}},
]

MODEL_TYPE = "mlp"
D_MODEL = 512
USE_REVIN = 1
SEED = 2024

results = []

for ds in DATASETS:
    for pred_len in [96, 192]:
        for m in MODELS:
            cfg_dict = dict(
                seq_len=ds["seq_len"], pred_len=pred_len, enc_in=ds["enc_in"], cycle=ds["cycle"],
                model_type=MODEL_TYPE, d_model=D_MODEL, use_revin=USE_REVIN,
            )
            cfg_dict.update(m["extra"])
            cfg = types.SimpleNamespace(**cfg_dict)

            model_id = f"{ds['name']}_{m['name']}_{ds['seq_len']}_{pred_len}"
            setting = (
                f"{model_id}_{m['name']}_{ds['data']}_ftM_sl{ds['seq_len']}_pl{pred_len}"
                f"_cycle{ds['cycle']}_{MODEL_TYPE}_seed{SEED}"
            )
            ckpt_path = f"./checkpoints/{setting}/checkpoint.pth"

            model = m["module"].Model(cfg)
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state)
            except FileNotFoundError:
                results.append((ds["name"], m["name"], pred_len, "checkpoint not found", "checkpoint not found"))
                continue
            except Exception as e:
                results.append((ds["name"], m["name"], pred_len, f"error: {e}", f"error: {e}"))
                continue

            trend_gate = f"{model.trend_branch.gate.item():.6f}" if m["has_trend"] else "n/a"
            mix_gate = f"{model.channel_mixing[0].gate.item():.6f}" if m["has_mix"] else "n/a"
            results.append((ds["name"], m["name"], pred_len, trend_gate, mix_gate))

# ---------------------------------------------------------------
# Print as a simple table
# ---------------------------------------------------------------
print(f"{'Dataset':10s} {'Model':20s} {'Horizon':8s} {'Trend gate':>14s} {'Mixing gate':>14s}")
print("=" * 70)
for row in results:
    print(f"{row[0]:10s} {row[1]:20s} {row[2]:<8d} {row[3]:>14s} {row[4]:>14s}")