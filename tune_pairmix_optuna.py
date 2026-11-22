# import argparse
# import gc
# import os
# import random

# import numpy as np
# import optuna
# import torch

# from exp.exp_main import Exp_Main


# parser = argparse.ArgumentParser(
#     description='Optuna TPE tuning for CycleNetPairMix'
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


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def build_args(trial=None, best_params=None):
#     args = argparse.Namespace()

#     args.random_seed = study_args.random_seed
#     args.is_training = 1
#     args.model = 'CycleNetPairMix'
#     args.model_id = 'optuna_pairmix'

#     args.data = study_args.data
#     args.root_path = study_args.root_path
#     args.data_path = study_args.data_path
#     args.features = 'M'
#     args.target = 'OT'
#     args.freq = 'h'
#     args.checkpoints = './checkpoints/'

#     args.seq_len = study_args.seq_len
#     args.label_len = 0
#     args.pred_len = study_args.pred_len

#     args.cycle = study_args.cycle
#     args.model_type = study_args.model_type
#     args.use_revin = 1

#     if trial is not None:
#         if study_args.data_name.lower() == 'weather':
#             args.mix_k = trial.suggest_int('mix_k', 1, 20)
#         elif study_args.data_name.lower() == 'ettm1':
#             args.mix_k = trial.suggest_int('mix_k', 1, 6)
#         else:
#             raise ValueError(f'Unknown dataset: {study_args.data_name}')

#         args.learning_rate = trial.suggest_float(
#             'learning_rate', 1e-5, 3e-3, log=True
#         )
#     else:
#         if best_params is None:
#             raise ValueError('best_params must be provided when trial is None')
#         args.mix_k = best_params['mix_k']
#         args.learning_rate = best_params['learning_rate']

#     if study_args.data_name.lower() == 'weather':
#         args.dependency_score_path = './dependency_scores/weather_dependency.npy'
#     elif study_args.data_name.lower() == 'ettm1':
#         args.dependency_score_path = './dependency_scores/ETTm1_dependency.npy'
#     else:
#         raise ValueError(f'Unknown dataset: {study_args.data_name}')

#     args.mix_ratio = 0.4
#     args.mix_hidden = 128
#     args.mix_layers = 1
#     args.mix_dropout = 0.1
#     args.mix_lag_window = 4
#     args.mix_rank = 16

#     args.trend_kernel = 25
#     args.trend_layers = 1

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

#     args.rnn_type = 'gru'
#     args.dec_way = 'pmf'
#     args.seg_len = 48
#     args.channel_id = 1
#     args.period_len = 24

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

#     args.use_gpu = torch.cuda.is_available()
#     args.gpu = 0
#     args.use_multi_gpu = False
#     args.devices = '0'
#     args.device_ids = [0]
#     args.test_flop = False

#     return args


# def make_setting(args, trial_number):
#     return (
#         f'{study_args.data_name}_CycleNetPairMix_'
#         f'{study_args.seq_len}_{study_args.pred_len}_'
#         f'trial{trial_number}_'
#         f'k{args.mix_k}_'
#         f'lr{args.learning_rate:.8f}'
#     )


# def objective(trial):
#     set_seed(study_args.random_seed)

#     args = build_args(trial=trial)
#     setting = make_setting(args, trial.number)

#     print()
#     print("======================================================")
#     print(f"OPTUNA TRIAL {trial.number}")
#     print("======================================================")
#     print(f"Dataset:          {study_args.data_name}")
#     print(f"Prediction:       {study_args.pred_len}")
#     print(f"mix_k:            {args.mix_k}")
#     print(f"learning_rate:    {args.learning_rate:.8f}")
#     print(f"dependency file:  {args.dependency_score_path}")
#     print("======================================================")

#     exp = Exp_Main(args)
#     result = exp.train(setting)

#     val_mse = float(result['val_mse'])
#     val_mae = float(result['val_mae'])
#     best_epoch = int(result['best_epoch'])

#     trial.set_user_attr('val_mae', val_mae)
#     trial.set_user_attr('best_epoch', best_epoch)
#     trial.set_user_attr('setting', setting)
#     trial.set_user_attr('dependency_score_path', args.dependency_score_path)

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

#     return val_mse


# def test_best_configuration(study):
#     best_trial = study.best_trial
#     best_params = study.best_params
#     best_setting = best_trial.user_attrs['setting']

#     print()
#     print("======================================================")
#     print("BEST CONFIGURATION - SELECTED BY VALIDATION MSE")
#     print("======================================================")
#     print(f"Dataset:        {study_args.data_name}")
#     print(f"Horizon:        {study_args.pred_len}")
#     print(f"Trial:          {best_trial.number}")
#     print()
#     print(f"mix_k:          {best_params['mix_k']}")
#     print(f"learning_rate:  {best_params['learning_rate']:.8f}")
#     print()
#     print(f"Validation MSE: {study.best_value:.8f}")
#     print(f"Validation MAE: {best_trial.user_attrs['val_mae']:.8f}")
#     print(f"Best epoch:     {best_trial.user_attrs['best_epoch']}")
#     print("======================================================")

#     args = build_args(best_params=best_params)
#     set_seed(study_args.random_seed)
#     exp = Exp_Main(args)

#     test_result = exp.test(best_setting, test=1)

#     print()
#     print("======================================================")
#     print("FINAL SELECTED MODEL")
#     print("======================================================")
#     print(f"Validation MSE: {study.best_value:.8f}")
#     print(f"Validation MAE: {best_trial.user_attrs['val_mae']:.8f}")
#     print(f"Test MSE:       {test_result['mse']:.8f}")
#     print(f"Test MAE:       {test_result['mae']:.8f}")
#     print("======================================================")

#     return test_result


# def main():
#     set_seed(study_args.random_seed)

#     os.makedirs('./optuna_results', exist_ok=True)

#     study_name = (
#         f'pairmix_'
#         f'{study_args.data_name}_'
#         f'h{study_args.pred_len}'
#     )

#     database_path = f'optuna_results/{study_name}.db'
#     csv_path = f'optuna_results/{study_name}.csv'

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
#     print("OPTUNA TPE STUDY - PAIRMIX")
#     print("======================================================")
#     print(f"Study:            {study_name}")
#     print(f"Dataset:          {study_args.data_name}")
#     print(f"Horizon:          {study_args.pred_len}")
#     print(f"Requested trials: {study_args.n_trials}")
#     print("Objective:         Validation MSE")
#     print("======================================================")

#     completed_trials = len([
#         t for t in study.trials
#         if t.state == optuna.trial.TrialState.COMPLETE
#     ])

#     remaining_trials = max(
#         0,
#         study_args.n_trials - completed_trials
#     )

#     print(f"Already completed: {completed_trials}")
#     print(f"Trials remaining:  {remaining_trials}")

#     if remaining_trials > 0:
#         study.optimize(
#             objective,
#             n_trials=remaining_trials,
#             gc_after_trial=True
#         )

#     study.trials_dataframe().to_csv(
#         csv_path,
#         index=False
#     )

#     print()
#     print(f"Trial table saved to: {csv_path}")
#     print(f"Study database:       {database_path}")

#     test_best_configuration(study)


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


parser = argparse.ArgumentParser(
    description='Validation-guided local Optuna TPE tuning for CycleNetPairMix'
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
#
# These are selected using VALIDATION performance, not test MSE.
#
# Weather has D = 21:
#   mix_ratio 0.4 -> k = ceil(0.4 * 20) = 8
#   mix_ratio 0.6 -> k = ceil(0.6 * 20) = 12
#
# ETTm1 has D = 7:
#   mix_ratio 0.2 -> k = ceil(0.2 * 6) = 2
#
PAIRMIX_PRIORS = {
    ('weather', 96): {
        'mix_k': 8,
        'learning_rate': 5e-4,
    },

    ('weather', 192): {
        'mix_k': 12,
        'learning_rate': 5e-4,
    },

    ('ettm1', 96): {
        'mix_k': 2,
        'learning_rate': 5e-4,
    },

    ('ettm1', 192): {
        'mix_k': 2,
        'learning_rate': 5e-4,
    },
}


# ============================================================
# LOCAL SEARCH SPACES
# ============================================================
#
# These ranges are centered around the promising regions
# identified by the normal-grid experiments, but are still
# wide enough to let Optuna move away from the grid winner.
#
PAIRMIX_SEARCH_SPACES = {
    ('weather', 96): {
        'k': (4, 12),
        'lr': (1e-4, 1.5e-3),
    },

    ('weather', 192): {
        'k': (6, 16),
        'lr': (1e-4, 1.5e-3),
    },

    ('ettm1', 96): {
        'k': (1, 4),
        'lr': (1e-4, 1.5e-3),
    },

    ('ettm1', 192): {
        'k': (1, 4),
        'lr': (1e-4, 1.5e-3),
    },
}


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_study_key():
    return (
        study_args.data_name.lower(),
        study_args.pred_len
    )


# ============================================================
# BUILD ARGUMENTS
# ============================================================

def build_args(trial=None, best_params=None):
    args = argparse.Namespace()

    # --------------------------------------------------------
    # BASIC CONFIG
    # --------------------------------------------------------

    args.random_seed = study_args.random_seed
    args.is_training = 1

    args.model = 'CycleNetPairMix'
    args.model_id = 'optuna_pairmix_local'

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
    # PAIRMIX HYPERPARAMETERS
    # --------------------------------------------------------

    if trial is not None:

        key = get_study_key()

        if key not in PAIRMIX_SEARCH_SPACES:
            raise ValueError(
                f'No local PairMix search space defined for {key}'
            )

        space = PAIRMIX_SEARCH_SPACES[key]

        args.mix_k = trial.suggest_int(
            'mix_k',
            space['k'][0],
            space['k'][1]
        )

        args.learning_rate = trial.suggest_float(
            'learning_rate',
            space['lr'][0],
            space['lr'][1],
            log=True
        )

    else:

        if best_params is None:
            raise ValueError(
                'best_params must be provided when trial is None'
            )

        args.mix_k = best_params['mix_k']
        args.learning_rate = best_params['learning_rate']

    # --------------------------------------------------------
    # PRECOMPUTED DEPENDENCY SCORES
    # --------------------------------------------------------
    #
    # Dependency scores are NOT recomputed during Optuna.
    # The same saved matrices from the earlier PairMix
    # experiments are reused.
    #

    if study_args.data_name.lower() == 'weather':

        args.dependency_score_path = (
            './dependency_scores/weather_dependency.npy'
        )

    elif study_args.data_name.lower() == 'ettm1':

        args.dependency_score_path = (
            './dependency_scores/ETTm1_dependency.npy'
        )

    else:

        raise ValueError(
            f'Unknown dataset: {study_args.data_name}'
        )

    # --------------------------------------------------------
    # PAIRMIX / MIXING DEFAULTS
    # --------------------------------------------------------

    args.mix_ratio = 0.4
    args.mix_hidden = 128
    args.mix_layers = 1
    args.mix_dropout = 0.1
    args.mix_lag_window = 4
    args.mix_rank = 16

    # --------------------------------------------------------
    # TREND DEFAULTS
    # --------------------------------------------------------

    args.trend_kernel = 25
    args.trend_layers = 1

    # --------------------------------------------------------
    # PATCHTST DEFAULTS
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
    # FORMER / GENERAL DEFAULTS
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

    # "Local" is included so checkpoint folders cannot clash
    # with the previous broad Optuna experiment.

    return (
        f'{study_args.data_name}_CycleNetPairMixLocal_'
        f'{study_args.seq_len}_{study_args.pred_len}_'
        f'trial{trial_number}_'
        f'k{args.mix_k}_'
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
    print(f"LOCAL OPTUNA TRIAL {trial.number}")
    print("======================================================")
    print(f"Dataset:          {study_args.data_name}")
    print(f"Prediction:       {study_args.pred_len}")
    print(f"mix_k:            {args.mix_k}")
    print(f"learning_rate:    {args.learning_rate:.8f}")
    print(f"dependency file:  {args.dependency_score_path}")
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

    # --------------------------------------------------------
    # SAVE TRIAL METADATA
    # --------------------------------------------------------

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

    trial.set_user_attr(
        'dependency_score_path',
        args.dependency_score_path
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
    # VALUE MINIMIZED BY TPE
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

    print(
        f"mix_k:          "
        f"{best_params['mix_k']}"
    )

    print(
        f"learning_rate:  "
        f"{best_params['learning_rate']:.8f}"
    )

    print()

    print(
        f"Validation MSE: "
        f"{study.best_value:.8f}"
    )

    print(
        f"Validation MAE: "
        f"{best_trial.user_attrs['val_mae']:.8f}"
    )

    print(
        f"Best epoch:     "
        f"{best_trial.user_attrs['best_epoch']}"
    )

    print("======================================================")

    # Build fresh configuration with selected parameters
    args = build_args(
        best_params=best_params
    )

    set_seed(
        study_args.random_seed
    )

    exp = Exp_Main(args)

    # --------------------------------------------------------
    # TEST ONLY THE VALIDATION-SELECTED WINNER
    # --------------------------------------------------------

    test_result = exp.test(
        best_setting,
        test=1
    )

    print()
    print("======================================================")
    print("FINAL SELECTED MODEL")
    print("======================================================")

    print(
        f"Validation MSE: "
        f"{study.best_value:.8f}"
    )

    print(
        f"Validation MAE: "
        f"{best_trial.user_attrs['val_mae']:.8f}"
    )

    print(
        f"Test MSE:       "
        f"{test_result['mse']:.8f}"
    )

    print(
        f"Test MAE:       "
        f"{test_result['mae']:.8f}"
    )

    print("======================================================")

    return test_result


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        study_args.random_seed
    )

    os.makedirs(
        './optuna_results',
        exist_ok=True
    )

    key = get_study_key()

    if key not in PAIRMIX_PRIORS:
        raise ValueError(
            f'No validation-selected PairMix prior defined for {key}'
        )

    if key not in PAIRMIX_SEARCH_SPACES:
        raise ValueError(
            f'No local PairMix search space defined for {key}'
        )

    # ========================================================
    # NEW STUDY NAME
    # ========================================================
    #
    # This must differ from the previous broad Optuna study.
    #

    study_name = (
        f'pairmix_local_'
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
    #
    # We already have a useful prior from the normal grid,
    # therefore we do not need 10 uninformed startup trials.
    #

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

    # ========================================================
    # ENQUEUE NORMAL-GRID WINNER
    # ========================================================
    #
    # This guarantees that the first trial in a fresh study
    # evaluates the best configuration already identified by
    # the validation-selected manual grid.
    #
    # If the study is being resumed, do NOT enqueue it again.
    #

    if len(study.trials) == 0:

        prior = PAIRMIX_PRIORS[key]

        print()
        print("======================================================")
        print("ENQUEUING NORMAL-GRID PRIOR")
        print("======================================================")
        print(f"Dataset:        {study_args.data_name}")
        print(f"Horizon:        {study_args.pred_len}")
        print(f"mix_k:          {prior['mix_k']}")

        print(
            f"learning_rate:  "
            f"{prior['learning_rate']:.8f}"
        )

        print("======================================================")

        study.enqueue_trial(
            prior
        )

    # ========================================================
    # STUDY INFORMATION
    # ========================================================

    print()
    print("======================================================")
    print("LOCAL OPTUNA TPE STUDY - PAIRMIX")
    print("======================================================")

    print(
        f"Study:            "
        f"{study_name}"
    )

    print(
        f"Dataset:          "
        f"{study_args.data_name}"
    )

    print(
        f"Horizon:          "
        f"{study_args.pred_len}"
    )

    print(
        f"Requested trials: "
        f"{study_args.n_trials}"
    )

    print(
        "Objective:         Validation MSE"
    )

    print(
        f"Local k range:     "
        f"{PAIRMIX_SEARCH_SPACES[key]['k']}"
    )

    print(
        f"Local LR range:    "
        f"{PAIRMIX_SEARCH_SPACES[key]['lr']}"
    )

    print("======================================================")

    # --------------------------------------------------------
    # RESUME STUDY CORRECTLY
    # --------------------------------------------------------

    completed_trials = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])

    remaining_trials = max(
        0,
        study_args.n_trials - completed_trials
    )

    print(
        f"Already completed: "
        f"{completed_trials}"
    )

    print(
        f"Trials remaining:  "
        f"{remaining_trials}"
    )

    # --------------------------------------------------------
    # RUN OPTUNA
    # --------------------------------------------------------

    if remaining_trials > 0:

        study.optimize(
            objective,
            n_trials=remaining_trials,
            gc_after_trial=True
        )

    # ========================================================
    # SAVE TRIAL RESULTS
    # ========================================================

    study.trials_dataframe().to_csv(
        csv_path,
        index=False
    )

    print()

    print(
        f"Trial table saved to: "
        f"{csv_path}"
    )

    print(
        f"Study database:       "
        f"{database_path}"
    )

    # ========================================================
    # TEST ONLY WINNING CONFIGURATION
    # ========================================================

    test_best_configuration(
        study
    )


if __name__ == '__main__':
    main()