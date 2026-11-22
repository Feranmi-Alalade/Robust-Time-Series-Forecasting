
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

SEQ_LEN = 96
PRED_LEN = 96
CYCLE = 144
ENC_IN = 137

DATA_PATH = Path("dataset/solar_AL.txt")

CHECKPOINT = Path(
    "checkpoints/"
    "Solar_Base_L96_H96_"
    "CycleNet_Solar_ftM_sl96_pl96_cycle144_mlp_seed2024/"
    "checkpoint.pth"
)

OUTPUT = Path("dependency_scores/Solar_dependency_L96.npy")


# ============================================================
# 1. Verify inputs
# ============================================================

if not DATA_PATH.is_file():
    raise FileNotFoundError(DATA_PATH)

if not CHECKPOINT.is_file():
    raise FileNotFoundError(CHECKPOINT)


# ============================================================
# 2. Load the Solar dataset
#    Follow CycleNet's original 70% training split
# ============================================================

data = np.loadtxt(
    DATA_PATH,
    delimiter=",",
    dtype=np.float32
)

if data.ndim != 2 or data.shape[1] != ENC_IN:
    raise ValueError(
        f"Expected Solar data with {ENC_IN} channels, "
        f"got {data.shape}"
    )

n_train = int(len(data) * 0.7)

train_data = data[:n_train]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

print("Full dataset shape:", data.shape)
print("Training data shape:", train_scaled.shape)


# ============================================================
# 3. Load the trained CycleNet cycle parameters
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

state = checkpoint

for wrapper_key in ("state_dict", "model_state_dict"):
    if isinstance(state, dict) and wrapper_key in state:
        state = state[wrapper_key]
        break

if not isinstance(state, dict):
    raise TypeError("Unexpected checkpoint format")

matching_keys = [
    key for key in state
    if key.endswith("cycleQueue.data")
]

if len(matching_keys) != 1:
    raise RuntimeError(
        "Could not uniquely identify cycleQueue.data. "
        f"Matching keys: {matching_keys}"
    )

cycle = state[matching_keys[0]].detach().cpu().numpy()

if cycle.shape != (CYCLE, ENC_IN):
    raise ValueError(
        f"Expected cycle shape {(CYCLE, ENC_IN)}, "
        f"got {cycle.shape}"
    )

print("Loaded learned cycle:", cycle.shape)


# ============================================================
# 4. Remove CycleNet's learned cycle
#
# Original CycleNet Solar loader passes the cycle index
# at the END of the input window.
#
# For an observation at absolute training index t,
# the corresponding input-cycle phase is:
#
#     (t + SEQ_LEN) % CYCLE
#
# We restrict observations to those covered by complete
# training input/target windows.
# ============================================================

last_input_exclusive = n_train - PRED_LEN

indices = np.arange(last_input_exclusive)

cycle_phases = (indices + SEQ_LEN) % CYCLE

residuals = (
    train_scaled[:last_input_exclusive]
    - cycle[cycle_phases]
)

print("Residual shape:", residuals.shape)


# ============================================================
# 5. Compute absolute Pearson correlation
# ============================================================

with np.errstate(invalid="ignore", divide="ignore"):
    correlation = np.corrcoef(
        residuals,
        rowvar=False
    )

dependency = np.abs(
    np.nan_to_num(
        correlation,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
).astype(np.float32)

np.fill_diagonal(dependency, 1.0)


# ============================================================
# 6. Validate and save
# ============================================================

assert dependency.shape == (ENC_IN, ENC_IN)
assert np.isfinite(dependency).all()
assert np.allclose(dependency, dependency.T, atol=1e-5)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

np.save(OUTPUT, dependency)

off_diag = dependency[~np.eye(ENC_IN, dtype=bool)]

print("\nDependency matrix generated successfully")
print("Path:", OUTPUT)
print("Shape:", dependency.shape)
print("Minimum:", dependency.min())
print("Maximum:", dependency.max())
print("Mean off-diagonal correlation:", off_diag.mean())
