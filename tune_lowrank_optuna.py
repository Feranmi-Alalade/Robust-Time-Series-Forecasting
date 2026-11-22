# import argparse
# import gc
# import os
# import random

# import numpy as np
# import optuna
# import torch

# from exp.exp_main import Exp_Main


# # ============================================================
# # OPTUNA COMMAND-LINE ARGUMENTS
# # ============================================================

# parser = argparse.ArgumentParser(
#     description='Optuna TPE tuning for CycleNetMixLowRank'
# )

# parser.add_argument('--data_name', type=str, required=True)
# parser.add_argument('--data_path', type=str, required=True)
# parser.add_argument('--data', type=str, required=True)
# parser.add_argument('--enc_in', type=int, required=True)
# parser.add_argument('--cycle', type=int, required=True)
# parser.add_argument('--seq_len', type=int, default=336)
# parser.add_argument('--pred_len', type=int, required=True)

# parser.add_argument('--root_path', type=str, default='./dataset/')
# parser.add_argument('--model_type', type=str, default='mlp')
# parser.add_argument('--batch_size', type=int, default=64)

# parser.add_argument('--train_epochs', type=int, default=30)
# parser.add_argument('--patience', type=int, default=5)
# parser.add_argument('--num_workers', type=int, default=10)

# parser.add_argument('--n_trials', type=int, default=30)
# parser.add_argument('--random_seed', type=int, default=2024)

# study_args = parser.parse_args()


# # ============================================================
# # RANDOM SEED
# # ============================================================

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# # ============================================================
# # BUILD ARGUMENTS EXPECTED BY YOUR EXISTING REPOSITORY
# # ============================================================

# def build_args(trial=None, best_params=None):

#     args = argparse.Namespace()

#     # --------------------------------------------------------
#     # BASIC CONFIG
#     # --------------------------------------------------------

#     args.random_seed = study_args.random_seed
#     args.is_training = 1

#     args.model = 'CycleNetMixLowRank'
#     args.model_id = 'optuna_lowrank'

#     # --------------------------------------------------------
#     # DATA
#     # --------------------------------------------------------

#     args.data = study_args.data
#     args.root_path = study_args.root_path
#     args.data_path = study_args.data_path
#     args.features = 'M'
#     args.target = 'OT'
#     args.freq = 'h'
#     args.checkpoints = './checkpoints/'

#     # --------------------------------------------------------
#     # FORECASTING
#     # --------------------------------------------------------

#     args.seq_len = study_args.seq_len
#     args.label_len = 0
#     args.pred_len = study_args.pred_len

#     # --------------------------------------------------------
#     # CYCLENET
#     # --------------------------------------------------------

#     args.cycle = study_args.cycle
#     args.model_type = study_args.model_type
#     args.use_revin = 1

#     # --------------------------------------------------------
#     # LOWRANK HYPERPARAMETERS
#     # --------------------------------------------------------

#     if trial is not None:

#         # True low-rank bottleneck:
#         # Weather has 21 channels
#         # ETTm1 has 7 channels

#         if study_args.data_name.lower() == 'weather':
#             args.mix_rank = trial.suggest_int(
#                 'mix_rank',
#                 2,
#                 20
#             )

#         elif study_args.data_name.lower() == 'ettm1':
#             args.mix_rank = trial.suggest_int(
#                 'mix_rank',
#                 1,
#                 6
#             )

#         else:
#             raise ValueError(
#                 f'Unknown dataset: {study_args.data_name}'
#             )

#         args.mix_layers = trial.suggest_int(
#             'mix_layers',
#             1,
#             3
#         )

#         args.mix_dropout = trial.suggest_float(
#             'mix_dropout',
#             0.0,
#             0.4
#         )

#         args.learning_rate = trial.suggest_float(
#             'learning_rate',
#             1e-5,
#             3e-3,
#             log=True
#         )

#     else:

#         args.mix_rank = best_params['mix_rank']
#         args.mix_layers = best_params['mix_layers']
#         args.mix_dropout = best_params['mix_dropout']
#         args.learning_rate = best_params['learning_rate']

#     # Other CycleNetMix arguments from run.py
#     args.mix_hidden = 128
#     args.mix_lag_window = 4

#     # PairMix defaults
#     args.dependency_score_path = None
#     args.mix_ratio = 0.4

#     # Trend defaults
#     args.trend_kernel = 25
#     args.trend_layers = 1

#     # --------------------------------------------------------
#     # PATCHTST DEFAULTS FROM run.py
#     # --------------------------------------------------------

#     args.fc_dropout = 0.05
#     args.head_dropout = 0.0
#     args.patch_len = 16
#     args.stride = 8
#     args.padding_patch = 'end'
#     args.revin = 0
#     args.affine = 0
#     args.subtract_last = 0
#     args.decomposition = 0
#     args.kernel_size = 25
#     args.individual = 0

#     # --------------------------------------------------------
#     # SEGRNN DEFAULTS
#     # --------------------------------------------------------

#     args.rnn_type = 'gru'
#     args.dec_way = 'pmf'
#     args.seg_len = 48
#     args.channel_id = 1

#     # SparseTSF
#     args.period_len = 24

#     # --------------------------------------------------------
#     # FORMER / GENERAL MODEL DEFAULTS
#     # --------------------------------------------------------

#     args.enc_in = study_args.enc_in
#     args.dec_in = study_args.enc_in
#     args.c_out = study_args.enc_in

#     args.embed_type = 0
#     args.d_model = 512
#     args.n_heads = 8
#     args.e_layers = 2
#     args.d_layers = 1
#     args.d_ff = 2048
#     args.moving_avg = 25
#     args.factor = 1

#     args.distil = True
#     args.dropout = 0
#     args.embed = 'timeF'
#     args.activation = 'gelu'

#     args.output_attention = False
#     args.do_predict = False

#     # --------------------------------------------------------
#     # OPTIMIZATION
#     # --------------------------------------------------------

#     args.num_workers = study_args.num_workers
#     args.itr = 1
#     args.train_epochs = study_args.train_epochs
#     args.batch_size = study_args.batch_size
#     args.patience = study_args.patience

#     args.des = 'optuna_tpe'
#     args.loss = 'mse'
#     args.lradj = 'type3'
#     args.pct_start = 0.3
#     args.use_amp = False

#     # --------------------------------------------------------
#     # GPU
#     # --------------------------------------------------------

#     args.use_gpu = torch.cuda.is_available()
#     args.gpu = 0
#     args.use_multi_gpu = False
#     args.devices = '0'
#     args.device_ids = [0]

#     args.test_flop = False

#     return args


# # ============================================================
# # SETTING NAME
# # ============================================================

# def make_setting(args, trial_number):

#     return (
#         f'{study_args.data_name}_CycleNetMixLowRank_'
#         f'{study_args.seq_len}_{study_args.pred_len}_'
#         f'trial{trial_number}_'
#         f'rank{args.mix_rank}_'
#         f'layers{args.mix_layers}_'
#         f'drop{args.mix_dropout:.4f}_'
#         f'lr{args.learning_rate:.8f}'
#     )


# # ============================================================
# # OPTUNA OBJECTIVE
# # ============================================================

# def objective(trial):

#     # Same initialization for every trial
#     set_seed(study_args.random_seed)

#     args = build_args(
#         trial=trial
#     )

#     setting = make_setting(
#         args,
#         trial.number
#     )

#     print()
#     print("======================================================")
#     print(f"OPTUNA TRIAL {trial.number}")
#     print("======================================================")
#     print(f"Dataset:          {study_args.data_name}")
#     print(f"Prediction:       {study_args.pred_len}")
#     print(f"mix_rank:         {args.mix_rank}")
#     print(f"mix_layers:       {args.mix_layers}")
#     print(f"mix_dropout:      {args.mix_dropout:.6f}")
#     print(f"learning_rate:    {args.learning_rate:.8f}")
#     print("======================================================")

#     exp = Exp_Main(args)

#     # IMPORTANT:
#     # No exp.test() here.
#     #
#     # train() must return metrics from the best
#     # validation checkpoint.
#     result = exp.train(setting)

#     val_mse = float(
#         result['val_mse']
#     )

#     val_mae = float(
#         result['val_mae']
#     )

#     best_epoch = int(
#         result['best_epoch']
#     )

#     # Save supplementary information with the trial
#     trial.set_user_attr(
#         'val_mae',
#         val_mae
#     )

#     trial.set_user_attr(
#         'best_epoch',
#         best_epoch
#     )

#     trial.set_user_attr(
#         'setting',
#         setting
#     )

#     print()
#     print("------------------------------------------------------")
#     print(f"Trial:          {trial.number}")
#     print(f"Best epoch:     {best_epoch}")
#     print(f"Validation MSE: {val_mse:.8f}")
#     print(f"Validation MAE: {val_mae:.8f}")
#     print("------------------------------------------------------")

#     del exp

#     gc.collect()

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # ========================================================
#     # THIS IS THE VALUE TPE MINIMIZES
#     # ========================================================

#     return val_mse


# # ============================================================
# # FINAL TEST
# # ============================================================

# def test_best_configuration(study):

#     best_trial = study.best_trial
#     best_params = study.best_params

#     best_setting = best_trial.user_attrs[
#         'setting'
#     ]

#     print()
#     print("======================================================")
#     print("BEST CONFIGURATION - SELECTED BY VALIDATION MSE")
#     print("======================================================")
#     print(f"Dataset:        {study_args.data_name}")
#     print(f"Horizon:        {study_args.pred_len}")
#     print(f"Trial:          {best_trial.number}")
#     print()
#     print(f"mix_rank:       {best_params['mix_rank']}")
#     print(f"mix_layers:     {best_params['mix_layers']}")
#     print(f"mix_dropout:    {best_params['mix_dropout']:.6f}")
#     print(f"learning_rate:  {best_params['learning_rate']:.8f}")
#     print()
#     print(f"Validation MSE: {study.best_value:.8f}")
#     print(
#         f"Validation MAE: "
#         f"{best_trial.user_attrs['val_mae']:.8f}"
#     )
#     print(
#         f"Best epoch:     "
#         f"{best_trial.user_attrs['best_epoch']}"
#     )
#     print("======================================================")

#     # Build a fresh model configuration
#     args = build_args(
#         best_params=best_params
#     )

#     set_seed(
#         study_args.random_seed
#     )

#     exp = Exp_Main(args)

#     # Load checkpoint belonging to winning trial
#     test_result = exp.test(
#         best_setting,
#         test=1
#     )

#     print()
#     print("======================================================")
#     print("FINAL SELECTED MODEL")
#     print("======================================================")
#     print(f"Validation MSE: {study.best_value:.8f}")
#     print(
#         f"Validation MAE: "
#         f"{best_trial.user_attrs['val_mae']:.8f}"
#     )
#     print(f"Test MSE:       {test_result['mse']:.8f}")
#     print(f"Test MAE:       {test_result['mae']:.8f}")
#     print("======================================================")

#     return test_result


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     set_seed(
#         study_args.random_seed
#     )

#     os.makedirs(
#         './optuna_results',
#         exist_ok=True
#     )

#     study_name = (
#         f'lowrank_'
#         f'{study_args.data_name}_'
#         f'h{study_args.pred_len}'
#     )

#     database_path = (
#         f'optuna_results/'
#         f'{study_name}.db'
#     )

#     csv_path = (
#         f'optuna_results/'
#         f'{study_name}.csv'
#     )

#     # ========================================================
#     # TPE SAMPLER
#     # ========================================================

#     sampler = optuna.samplers.TPESampler(
#         seed=study_args.random_seed,
#         n_startup_trials=10
#     )

#     study = optuna.create_study(
#         study_name=study_name,
#         direction='minimize',
#         sampler=sampler,
#         storage=f'sqlite:///{database_path}',
#         load_if_exists=True
#     )

#     print()
#     print("======================================================")
#     print("OPTUNA TPE STUDY")
#     print("======================================================")
#     print(f"Study:            {study_name}")
#     print(f"Dataset:          {study_args.data_name}")
#     print(f"Horizon:          {study_args.pred_len}")
#     print(f"Requested trials: {study_args.n_trials}")
#     print("Objective:         Validation MSE")
#     print("======================================================")

#     # --------------------------------------------------------
#     # Resume correctly if some trials already exist
#     # --------------------------------------------------------

#     completed_trials = len([
#         t for t in study.trials
#         if t.state == optuna.trial.TrialState.COMPLETE
#     ])

#     remaining_trials = max(
#         0,
#         study_args.n_trials - completed_trials
#     )

#     print(
#         f"Already completed: {completed_trials}"
#     )

#     print(
#         f"Trials remaining:  {remaining_trials}"
#     )

#     if remaining_trials > 0:

#         study.optimize(
#             objective,
#             n_trials=remaining_trials,
#             gc_after_trial=True
#         )

#     # ========================================================
#     # SAVE ALL TRIAL RESULTS
#     # ========================================================

#     study.trials_dataframe().to_csv(
#         csv_path,
#         index=False
#     )

#     print()
#     print(
#         f"Trial table saved to: {csv_path}"
#     )

#     print(
#         f"Study database:       {database_path}"
#     )

#     # ========================================================
#     # TEST ONLY WINNING CONFIGURATION
#     # ========================================================

#     test_best_configuration(
#         study
#     )


# if __name__ == '__main__':
#     main()

import argparse
import gc
import os
import random

import numpy as np
import optuna
import torch

from exp.exp_main import Exp_Main


# ============================================================
# OPTUNA COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        'Validation-guided local Optuna TPE tuning for '
        'CycleNetMixLowRank'
    )
)

parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--enc_in', type=int, required=True)
parser.add_argument('--cycle', type=int, required=True)
parser.add_argument('--seq_len', type=int, default=336)
parser.add_argument('--pred_len', type=int, required=True)

parser.add_argument('--root_path', type=str, default='./dataset/')
parser.add_argument('--model_type', type=str, default='mlp')
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--train_epochs', type=int, default=30)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=10)

parser.add_argument('--n_trials', type=int, default=30)
parser.add_argument('--random_seed', type=int, default=2024)

study_args = parser.parse_args()


# ============================================================
# VALIDATION-SELECTED NORMAL-GRID PRIORS
# ============================================================
# These are the configurations selected by minimum validation loss
# in the earlier manual grid search. They are used only as prior
# information for the new local Optuna study; test performance is
# not used to define the search.

LOWRANK_PRIORS = {
    ('weather', 96): {
        'mix_rank': 4,
        'mix_layers': 1,
        'mix_dropout': 0.1,
        'learning_rate': 1e-3,
    },
    ('weather', 192): {
        'mix_rank': 8,
        'mix_layers': 2,
        'mix_dropout': 0.1,
        'learning_rate': 5e-4,
    },
    ('ettm1', 96): {
        'mix_rank': 2,
        'mix_layers': 2,
        'mix_dropout': 0.1,
        'learning_rate': 1e-4,
    },
    ('ettm1', 192): {
        'mix_rank': 4,
        'mix_layers': 2,
        'mix_dropout': 0.1,
        'learning_rate': 1e-4,
    },
}


# ============================================================
# LOCAL SEARCH SPACES AROUND THE NORMAL-GRID WINNERS
# ============================================================
# Rank and learning-rate ranges are narrowed around the promising
# region found by the manual grid. Layers and dropout remain broad
# enough to let Optuna discover configurations not present in the
# original grid.

LOWRANK_SEARCH_SPACES = {
    ('weather', 96): {
        'rank': (2, 8),
        'lr': (3e-4, 2e-3),
    },
    ('weather', 192): {
        'rank': (4, 12),
        'lr': (1e-4, 1.5e-3),
    },
    ('ettm1', 96): {
        'rank': (1, 4),
        'lr': (3e-5, 5e-4),
    },
    ('ettm1', 192): {
        'rank': (2, 6),
        'lr': (3e-5, 5e-4),
    },
}


def study_key():
    return (
        study_args.data_name.lower(),
        study_args.pred_len,
    )


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# BUILD ARGUMENTS EXPECTED BY YOUR EXISTING REPOSITORY
# ============================================================

def build_args(trial=None, best_params=None):

    args = argparse.Namespace()

    # --------------------------------------------------------
    # BASIC CONFIG
    # --------------------------------------------------------

    args.random_seed = study_args.random_seed
    args.is_training = 1

    args.model = 'CycleNetMixLowRank'
    args.model_id = 'optuna_lowrank_local'

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    args.data = study_args.data
    args.root_path = study_args.root_path
    args.data_path = study_args.data_path
    args.features = 'M'
    args.target = 'OT'
    args.freq = 'h'
    args.checkpoints = './checkpoints/'

    # --------------------------------------------------------
    # FORECASTING
    # --------------------------------------------------------

    args.seq_len = study_args.seq_len
    args.label_len = 0
    args.pred_len = study_args.pred_len

    # --------------------------------------------------------
    # CYCLENET
    # --------------------------------------------------------

    args.cycle = study_args.cycle
    args.model_type = study_args.model_type
    args.use_revin = 1

    # --------------------------------------------------------
    # LOWRANK HYPERPARAMETERS
    # --------------------------------------------------------

    if trial is not None:

        key = study_key()

        if key not in LOWRANK_SEARCH_SPACES:
            raise ValueError(
                f'No local LowRank search space defined for {key}'
            )

        space = LOWRANK_SEARCH_SPACES[key]

        args.mix_rank = trial.suggest_int(
            'mix_rank',
            space['rank'][0],
            space['rank'][1],
        )

        # Keep depth slightly broader than the old {1, 2} grid.
        args.mix_layers = trial.suggest_int(
            'mix_layers',
            1,
            3,
        )

        # Dropout was fixed at 0.1 in normal tuning, so retain a
        # reasonably broad interval rather than forcing values near 0.1.
        args.mix_dropout = trial.suggest_float(
            'mix_dropout',
            0.0,
            0.4,
        )

        args.learning_rate = trial.suggest_float(
            'learning_rate',
            space['lr'][0],
            space['lr'][1],
            log=True,
        )

    else:

        if best_params is None:
            raise ValueError(
                'best_params must be provided when trial is None'
            )

        args.mix_rank = best_params['mix_rank']
        args.mix_layers = best_params['mix_layers']
        args.mix_dropout = best_params['mix_dropout']
        args.learning_rate = best_params['learning_rate']

    # Other CycleNetMix arguments from run.py
    args.mix_hidden = 128
    args.mix_lag_window = 4

    # PairMix defaults
    args.dependency_score_path = None
    args.mix_ratio = 0.4

    # Trend defaults
    args.trend_kernel = 25
    args.trend_layers = 1

    # --------------------------------------------------------
    # PATCHTST DEFAULTS FROM run.py
    # --------------------------------------------------------

    args.fc_dropout = 0.05
    args.head_dropout = 0.0
    args.patch_len = 16
    args.stride = 8
    args.padding_patch = 'end'
    args.revin = 0
    args.affine = 0
    args.subtract_last = 0
    args.decomposition = 0
    args.kernel_size = 25
    args.individual = 0

    # --------------------------------------------------------
    # SEGRNN DEFAULTS
    # --------------------------------------------------------

    args.rnn_type = 'gru'
    args.dec_way = 'pmf'
    args.seg_len = 48
    args.channel_id = 1

    # SparseTSF
    args.period_len = 24

    # --------------------------------------------------------
    # FORMER / GENERAL MODEL DEFAULTS
    # --------------------------------------------------------

    args.enc_in = study_args.enc_in
    args.dec_in = study_args.enc_in
    args.c_out = study_args.enc_in

    args.embed_type = 0
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.d_ff = 2048
    args.moving_avg = 25
    args.factor = 1

    args.distil = True
    args.dropout = 0
    args.embed = 'timeF'
    args.activation = 'gelu'

    args.output_attention = False
    args.do_predict = False

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    args.num_workers = study_args.num_workers
    args.itr = 1
    args.train_epochs = study_args.train_epochs
    args.batch_size = study_args.batch_size
    args.patience = study_args.patience

    args.des = 'optuna_tpe_local'
    args.loss = 'mse'
    args.lradj = 'type3'
    args.pct_start = 0.3
    args.use_amp = False

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0'
    args.device_ids = [0]

    args.test_flop = False

    return args


# ============================================================
# SETTING NAME
# ============================================================

def make_setting(args, trial_number):

    return (
        f'{study_args.data_name}_CycleNetMixLowRank_LOCAL_'
        f'{study_args.seq_len}_{study_args.pred_len}_'
        f'trial{trial_number}_'
        f'rank{args.mix_rank}_'
        f'layers{args.mix_layers}_'
        f'drop{args.mix_dropout:.4f}_'
        f'lr{args.learning_rate:.8f}'
    )


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    # Same initialization for every trial
    set_seed(study_args.random_seed)

    args = build_args(
        trial=trial
    )

    setting = make_setting(
        args,
        trial.number
    )

    print()
    print("======================================================")
    print(f"OPTUNA LOCAL TRIAL {trial.number}")
    print("======================================================")
    print(f"Dataset:          {study_args.data_name}")
    print(f"Prediction:       {study_args.pred_len}")
    print(f"mix_rank:         {args.mix_rank}")
    print(f"mix_layers:       {args.mix_layers}")
    print(f"mix_dropout:      {args.mix_dropout:.6f}")
    print(f"learning_rate:    {args.learning_rate:.8f}")
    print("======================================================")

    exp = Exp_Main(args)

    # IMPORTANT:
    # No exp.test() here.
    #
    # train() must return metrics from the best
    # validation checkpoint.
    result = exp.train(setting)

    val_mse = float(
        result['val_mse']
    )

    val_mae = float(
        result['val_mae']
    )

    best_epoch = int(
        result['best_epoch']
    )

    # Save supplementary information with the trial
    trial.set_user_attr(
        'val_mae',
        val_mae
    )

    trial.set_user_attr(
        'best_epoch',
        best_epoch
    )

    trial.set_user_attr(
        'setting',
        setting
    )

    print()
    print("------------------------------------------------------")
    print(f"Trial:          {trial.number}")
    print(f"Best epoch:     {best_epoch}")
    print(f"Validation MSE: {val_mse:.8f}")
    print(f"Validation MAE: {val_mae:.8f}")
    print("------------------------------------------------------")

    del exp

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # THIS IS THE VALUE TPE MINIMIZES
    # ========================================================

    return val_mse


# ============================================================
# FINAL TEST
# ============================================================

def test_best_configuration(study):

    best_trial = study.best_trial
    best_params = study.best_params

    best_setting = best_trial.user_attrs[
        'setting'
    ]

    print()
    print("======================================================")
    print("BEST CONFIGURATION - SELECTED BY VALIDATION MSE")
    print("======================================================")
    print(f"Dataset:        {study_args.data_name}")
    print(f"Horizon:        {study_args.pred_len}")
    print(f"Trial:          {best_trial.number}")
    print()
    print(f"mix_rank:       {best_params['mix_rank']}")
    print(f"mix_layers:     {best_params['mix_layers']}")
    print(f"mix_dropout:    {best_params['mix_dropout']:.6f}")
    print(f"learning_rate:  {best_params['learning_rate']:.8f}")
    print()
    print(f"Validation MSE: {study.best_value:.8f}")
    print(
        f"Validation MAE: "
        f"{best_trial.user_attrs['val_mae']:.8f}"
    )
    print(
        f"Best epoch:     "
        f"{best_trial.user_attrs['best_epoch']}"
    )
    print("======================================================")

    # Build a fresh model configuration
    args = build_args(
        best_params=best_params
    )

    set_seed(
        study_args.random_seed
    )

    exp = Exp_Main(args)

    # Load checkpoint belonging to winning trial
    test_result = exp.test(
        best_setting,
        test=1
    )

    print()
    print("======================================================")
    print("FINAL SELECTED MODEL")
    print("======================================================")
    print(f"Validation MSE: {study.best_value:.8f}")
    print(
        f"Validation MAE: "
        f"{best_trial.user_attrs['val_mae']:.8f}"
    )
    print(f"Test MSE:       {test_result['mse']:.8f}")
    print(f"Test MAE:       {test_result['mae']:.8f}")
    print("======================================================")

    return test_result


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        study_args.random_seed
    )

    key = study_key()

    if key not in LOWRANK_PRIORS:
        raise ValueError(
            f'No validation-selected LowRank prior defined for {key}'
        )

    if key not in LOWRANK_SEARCH_SPACES:
        raise ValueError(
            f'No local LowRank search space defined for {key}'
        )

    os.makedirs(
        './optuna_results',
        exist_ok=True
    )

    # IMPORTANT: use a new study name so the old guided-search
    # database is not loaded by load_if_exists=True.
    study_name = (
        f'lowrank_local_v2_'
        f'{study_args.data_name}_'
        f'h{study_args.pred_len}'
    )

    database_path = (
        f'optuna_results/'
        f'{study_name}.db'
    )

    csv_path = (
        f'optuna_results/'
        f'{study_name}.csv'
    )

    # ========================================================
    # TPE SAMPLER
    # ========================================================

    # Fewer purely random startup trials are needed because the
    # first queued trial is already an informed manual-grid winner.
    sampler = optuna.samplers.TPESampler(
        seed=study_args.random_seed,
        n_startup_trials=3
    )

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        storage=f'sqlite:///{database_path}',
        load_if_exists=True
    )

    # Seed a brand-new local study with the validation-selected
    # winner from the earlier normal grid. This configuration is
    # evaluated again under the same Optuna objective so that it is
    # part of the study history on equal footing with all new trials.
    if len(study.trials) == 0:
        prior = LOWRANK_PRIORS[key]

        print()
        print("======================================================")
        print("ENQUEUING VALIDATION-SELECTED NORMAL-GRID WINNER")
        print("======================================================")
        print(f"Dataset:          {study_args.data_name}")
        print(f"Horizon:          {study_args.pred_len}")
        print(f"mix_rank:         {prior['mix_rank']}")
        print(f"mix_layers:       {prior['mix_layers']}")
        print(f"mix_dropout:      {prior['mix_dropout']}")
        print(f"learning_rate:    {prior['learning_rate']}")
        print("======================================================")

        study.enqueue_trial(prior)

    print()
    print("======================================================")
    print("OPTUNA TPE STUDY - VALIDATION-GUIDED LOCAL REFINEMENT")
    print("======================================================")
    print(f"Study:            {study_name}")
    print(f"Dataset:          {study_args.data_name}")
    print(f"Horizon:          {study_args.pred_len}")
    print(f"Requested trials: {study_args.n_trials}")
    print("Objective:         Validation MSE")
    print(f"Local rank range:  {LOWRANK_SEARCH_SPACES[key]['rank']}")
    print(f"Local LR range:    {LOWRANK_SEARCH_SPACES[key]['lr']}")
    print("======================================================")

    # --------------------------------------------------------
    # Resume correctly if some trials already exist
    # --------------------------------------------------------

    completed_trials = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])

    running_or_waiting_trials = len([
        t for t in study.trials
        if t.state in (
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.WAITING,
        )
    ])

    # WAITING includes the queued normal-grid prior. Count it
    # toward the requested total so --n_trials=30 still means
    # 30 total study trials, not 31.
    existing_trials = completed_trials + running_or_waiting_trials

    remaining_trials = max(
        0,
        study_args.n_trials - existing_trials
    )

    print(
        f"Completed trials: {completed_trials}"
    )
    print(
        f"Queued/running:   {running_or_waiting_trials}"
    )
    print(
        f"Trials remaining: {remaining_trials}"
    )

    # If there is an enqueued WAITING trial, study.optimize must be
    # called enough times to execute it. The requested number here
    # includes all remaining new trials after accounting for WAITING.
    trials_to_run = remaining_trials + running_or_waiting_trials

    if trials_to_run > 0:

        study.optimize(
            objective,
            n_trials=trials_to_run,
            gc_after_trial=True
        )

    # ========================================================
    # SAVE ALL TRIAL RESULTS
    # ========================================================

    study.trials_dataframe().to_csv(
        csv_path,
        index=False
    )

    print()
    print(
        f"Trial table saved to: {csv_path}"
    )

    print(
        f"Study database:       {database_path}"
    )

    # ========================================================
    # TEST ONLY WINNING CONFIGURATION
    # ========================================================

    test_best_configuration(
        study
    )


if __name__ == '__main__':
    main()