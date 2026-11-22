import argparse
import gc
import os
import random

import numpy as np
import optuna
import torch

from exp.exp_main import Exp_Main


parser = argparse.ArgumentParser(
    description="Broad Optuna TPE tuning for CycleNetTrend"
)

parser.add_argument("--data_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--enc_in", type=int, required=True)
parser.add_argument("--cycle", type=int, required=True)
parser.add_argument("--seq_len", type=int, default=336)
parser.add_argument("--pred_len", type=int, required=True)
parser.add_argument("--root_path", type=str, default="./dataset/")
parser.add_argument("--model_type", type=str, default="mlp")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--train_epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument("--n_trials", type=int, default=50)
parser.add_argument("--random_seed", type=int, default=2024)

study_args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args(trial=None, best_params=None):
    args = argparse.Namespace()

    args.random_seed = study_args.random_seed
    args.is_training = 1
    args.model = "CycleNetTrend"
    args.model_id = "optuna_trend_broad"

    args.data = study_args.data
    args.root_path = study_args.root_path
    args.data_path = study_args.data_path
    args.features = "M"
    args.target = "OT"
    args.freq = "h"
    args.checkpoints = "./checkpoints/"

    args.seq_len = study_args.seq_len
    args.label_len = 0
    args.pred_len = study_args.pred_len

    args.cycle = study_args.cycle
    args.model_type = study_args.model_type
    args.use_revin = 1

    if trial is not None:
        args.trend_kernel = trial.suggest_int(
            "trend_kernel",
            5,
            193,
            step=2
        )

        args.trend_layers = trial.suggest_int(
            "trend_layers",
            1,
            2
        )

        args.learning_rate = trial.suggest_float(
            "learning_rate",
            1e-5,
            3e-3,
            log=True
        )
    else:
        if best_params is None:
            raise ValueError(
                "best_params must be provided when trial is None"
            )

        args.trend_kernel = best_params["trend_kernel"]
        args.trend_layers = best_params["trend_layers"]
        args.learning_rate = best_params["learning_rate"]

    args.mix_hidden = 128
    args.mix_layers = 1
    args.mix_dropout = 0.1
    args.mix_lag_window = 4
    args.mix_rank = 16
    args.dependency_score_path = None
    args.mix_ratio = 0.4

    args.fc_dropout = 0.05
    args.head_dropout = 0.0
    args.patch_len = 16
    args.stride = 8
    args.padding_patch = "end"
    args.revin = 0
    args.affine = 0
    args.subtract_last = 0
    args.decomposition = 0
    args.kernel_size = 25
    args.individual = 0

    args.rnn_type = "gru"
    args.dec_way = "pmf"
    args.seg_len = 48
    args.channel_id = 1
    args.period_len = 24

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
    args.embed = "timeF"
    args.activation = "gelu"
    args.output_attention = False
    args.do_predict = False

    args.num_workers = study_args.num_workers
    args.itr = 1
    args.train_epochs = study_args.train_epochs
    args.batch_size = study_args.batch_size
    args.patience = study_args.patience
    args.des = "optuna_trend_broad"
    args.loss = "mse"
    args.lradj = "type3"
    args.pct_start = 0.3
    args.use_amp = False

    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = "0"
    args.device_ids = [0]
    args.test_flop = False

    return args


def make_setting(args, trial_number):
    return (
        f"{study_args.data_name}_"
        f"CycleNetTrend_BROAD_"
        f"sl{study_args.seq_len}_"
        f"pl{study_args.pred_len}_"
        f"trial{trial_number}_"
        f"kernel{args.trend_kernel}_"
        f"layers{args.trend_layers}_"
        f"lr{args.learning_rate:.8f}_"
        f"seed{study_args.random_seed}"
    )


def objective(trial):
    set_seed(study_args.random_seed)

    args = build_args(trial=trial)
    setting = make_setting(args, trial.number)

    print()
    print("======================================================")
    print(f"OPTUNA BROAD TREND TRIAL {trial.number}")
    print("======================================================")
    print(f"Dataset:          {study_args.data_name}")
    print(f"Prediction:       {study_args.pred_len}")
    print(f"Lookback:         {study_args.seq_len}")
    print(f"Cycle:            {study_args.cycle}")
    print(f"trend_kernel:     {args.trend_kernel}")
    print(f"trend_layers:     {args.trend_layers}")
    print(f"learning_rate:    {args.learning_rate:.8f}")
    print("======================================================")

    exp = None

    try:
        exp = Exp_Main(args)

        result = exp.train(setting)

        val_mse = float(result["val_mse"])
        val_mae = float(result["val_mae"])
        best_epoch = int(result["best_epoch"])

        trial.set_user_attr("val_mae", val_mae)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("setting", setting)

        print()
        print("------------------------------------------------------")
        print(f"Trial:          {trial.number}")
        print(f"Best epoch:     {best_epoch}")
        print(f"Validation MSE: {val_mse:.8f}")
        print(f"Validation MAE: {val_mae:.8f}")
        print("------------------------------------------------------")

        return val_mse

    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print()
            print("CUDA OUT OF MEMORY - PRUNING TRIAL")
            raise optuna.TrialPruned()

        raise

    finally:
        if exp is not None:
            del exp

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_best_configuration(study):
    best_trial = study.best_trial
    best_params = study.best_params
    best_setting = best_trial.user_attrs["setting"]

    print()
    print("======================================================")
    print("BEST CONFIGURATION - SELECTED BY VALIDATION MSE")
    print("======================================================")
    print(f"Dataset:        {study_args.data_name}")
    print(f"Horizon:        {study_args.pred_len}")
    print(f"Trial:          {best_trial.number}")
    print()
    print(f"trend_kernel:   {best_params['trend_kernel']}")
    print(f"trend_layers:   {best_params['trend_layers']}")
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

    args = build_args(best_params=best_params)

    set_seed(study_args.random_seed)

    exp = Exp_Main(args)

    test_result = exp.test(
        best_setting,
        test=1
    )

    print()
    print("======================================================")
    print("FINAL SELECTED CYCLENETTREND MODEL")
    print("======================================================")
    print(f"trend_kernel:   {best_params['trend_kernel']}")
    print(f"trend_layers:   {best_params['trend_layers']}")
    print(f"learning_rate:  {best_params['learning_rate']:.8f}")
    print()
    print(f"Validation MSE: {study.best_value:.8f}")
    print(
        f"Validation MAE: "
        f"{best_trial.user_attrs['val_mae']:.8f}"
    )
    print(f"Test MSE:       {test_result['mse']:.8f}")
    print(f"Test MAE:       {test_result['mae']:.8f}")
    print("======================================================")

    del exp

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_result


def main():
    set_seed(study_args.random_seed)

    os.makedirs(
        "./optuna_results",
        exist_ok=True
    )

    study_name = (
        f"trend_broad_v1_"
        f"{study_args.data_name}_"
        f"h{study_args.pred_len}"
    )

    database_path = (
        f"optuna_results/"
        f"{study_name}.db"
    )

    csv_path = (
        f"optuna_results/"
        f"{study_name}.csv"
    )

    sampler = optuna.samplers.TPESampler(
        seed=study_args.random_seed,
        n_startup_trials=8
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{database_path}",
        load_if_exists=True
    )

    print()
    print("======================================================")
    print("CYCLENETTREND - BROAD OPTUNA TPE SEARCH")
    print("======================================================")
    print(f"Study:            {study_name}")
    print(f"Dataset:          {study_args.data_name}")
    print(f"Horizon:          {study_args.pred_len}")
    print(f"Lookback:         {study_args.seq_len}")
    print(f"Cycle:            {study_args.cycle}")
    print(f"Requested trials: {study_args.n_trials}")
    print("Objective:         Validation MSE")
    print("Kernel range:      5 - 193 (odd values)")
    print("Trend layers:      1 - 2")
    print("Learning rate:     1e-5 - 3e-3")
    print("======================================================")

    completed_trials = len([
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])

    print(f"Completed trials: {completed_trials}")

    remaining_trials = max(
        0,
        study_args.n_trials - completed_trials
    )

    print(f"Trials remaining: {remaining_trials}")

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            gc_after_trial=True
        )

    study.trials_dataframe().to_csv(
        csv_path,
        index=False
    )

    print()
    print(f"Trial table saved to: {csv_path}")
    print(f"Study database:       {database_path}")

    print()
    print("======================================================")
    print("OPTUNA SEARCH COMPLETE")
    print("======================================================")
    print(f"Best trial:      {study.best_trial.number}")
    print(f"Best Val MSE:    {study.best_value:.8f}")
    print(f"Best parameters: {study.best_params}")
    print("======================================================")

    test_best_configuration(study)


if __name__ == "__main__":
    main()