import argparse
import random

import numpy as np
import torch

from exp.exp_main import Exp_Main


parser = argparse.ArgumentParser(
    description='Model family for Time Series Forecasting'
)


# ============================================================
# RANDOM SEED
# ============================================================

parser.add_argument(
    '--random_seed',
    type=int,
    default=2024,
    help='random seed'
)


# ============================================================
# BASIC CONFIG
# ============================================================

parser.add_argument(
    '--is_training',
    type=int,
    required=True,
    default=1,
    help='status'
)

parser.add_argument(
    '--model_id',
    type=str,
    required=True,
    default='test',
    help='model id'
)

parser.add_argument(
    '--model',
    type=str,
    required=True,
    default='Autoformer',
    help='model name'
)


# ============================================================
# DATA LOADER
# ============================================================

parser.add_argument(
    '--data',
    type=str,
    required=True,
    default='ETTh1',
    help='dataset type'
)

parser.add_argument(
    '--root_path',
    type=str,
    default='./data/ETT/',
    help='root path of the data file'
)

parser.add_argument(
    '--data_path',
    type=str,
    default='ETTh1.csv',
    help='data file'
)

parser.add_argument(
    '--features',
    type=str,
    default='M',
    help='forecasting task: M, S, MS'
)

parser.add_argument(
    '--target',
    type=str,
    default='OT',
    help='target feature in S or MS task'
)

parser.add_argument(
    '--freq',
    type=str,
    default='h',
    help='frequency for time feature encoding'
)

parser.add_argument(
    '--checkpoints',
    type=str,
    default='./checkpoints/',
    help='location of model checkpoints'
)


# ============================================================
# FORECASTING TASK
# ============================================================

parser.add_argument(
    '--seq_len',
    type=int,
    default=96,
    help='input sequence length'
)

parser.add_argument(
    '--label_len',
    type=int,
    default=0,
    help='start token length'
)

parser.add_argument(
    '--pred_len',
    type=int,
    default=96,
    help='prediction sequence length'
)


# ============================================================
# CYCLENET
# ============================================================

parser.add_argument(
    '--cycle',
    type=int,
    default=24,
    help='cycle length'
)

parser.add_argument(
    '--model_type',
    type=str,
    default='mlp',
    help='model type: linear or mlp'
)

parser.add_argument(
    '--use_revin',
    type=int,
    default=1,
    help='1: use RevIN, 0: no RevIN'
)


# ============================================================
# CYCLENET MIX
# ============================================================

parser.add_argument(
    '--mix_hidden',
    type=int,
    default=128,
    help='hidden width of channel-mixing MLP'
)

parser.add_argument(
    '--mix_layers',
    type=int,
    default=1,
    help='number of stacked channel-mixing blocks'
)

parser.add_argument(
    '--mix_dropout',
    type=float,
    default=0.1,
    help='dropout inside channel-mixing MLP'
)

parser.add_argument(
    '--mix_lag_window',
    type=int,
    default=4,
    help='lag window for CycleNetMixLag'
)

parser.add_argument(
    '--mix_rank',
    type=int,
    default=16,
    help='low-rank dimension for CycleNetMixLowRank'
)


# ============================================================
# CYCLENET PAIR MIX
# ============================================================

parser.add_argument(
    '--dependency_score_path',
    type=str,
    default=None,
    help='path to D x D dependency score matrix for CycleNetPairMix'
)

parser.add_argument(
    '--mix_ratio',
    type=float,
    default=0.4,
    help='proportion of other channels selected for each target channel'
)


# ============================================================
# PATCHTST
# ============================================================

parser.add_argument(
    '--fc_dropout',
    type=float,
    default=0.05,
    help='fully connected dropout'
)

parser.add_argument(
    '--head_dropout',
    type=float,
    default=0.0,
    help='head dropout'
)

parser.add_argument(
    '--patch_len',
    type=int,
    default=16,
    help='patch length'
)

parser.add_argument(
    '--stride',
    type=int,
    default=8,
    help='stride'
)

parser.add_argument(
    '--padding_patch',
    default='end',
    help='padding on the end'
)

parser.add_argument(
    '--revin',
    type=int,
    default=0,
    help='RevIN'
)

parser.add_argument(
    '--affine',
    type=int,
    default=0,
    help='RevIN affine'
)

parser.add_argument(
    '--subtract_last',
    type=int,
    default=0,
    help='0: subtract mean; 1: subtract last'
)

parser.add_argument(
    '--decomposition',
    type=int,
    default=0,
    help='decomposition'
)

parser.add_argument(
    '--kernel_size',
    type=int,
    default=25,
    help='decomposition kernel'
)

parser.add_argument(
    '--individual',
    type=int,
    default=0,
    help='individual head'
)


# ============================================================
# SEGRNN
# ============================================================

parser.add_argument(
    '--rnn_type',
    default='gru',
    help='RNN type'
)

parser.add_argument(
    '--dec_way',
    default='pmf',
    help='decode way'
)

parser.add_argument(
    '--seg_len',
    type=int,
    default=48,
    help='segment length'
)

parser.add_argument(
    '--channel_id',
    type=int,
    default=1,
    help='enable channel position encoding'
)


# ============================================================
# SPARSE TSF
# ============================================================

parser.add_argument(
    '--period_len',
    type=int,
    default=24,
    help='period length'
)


# ============================================================
# FORMERS
# ============================================================

parser.add_argument(
    '--embed_type',
    type=int,
    default=0,
    help='embedding type'
)

parser.add_argument(
    '--enc_in',
    type=int,
    default=7,
    help='encoder input size'
)

parser.add_argument(
    '--dec_in',
    type=int,
    default=7,
    help='decoder input size'
)

parser.add_argument(
    '--c_out',
    type=int,
    default=7,
    help='output size'
)

parser.add_argument(
    '--d_model',
    type=int,
    default=512,
    help='dimension of model'
)

parser.add_argument(
    '--n_heads',
    type=int,
    default=8,
    help='number of heads'
)

parser.add_argument(
    '--e_layers',
    type=int,
    default=2,
    help='number of encoder layers'
)

parser.add_argument(
    '--d_layers',
    type=int,
    default=1,
    help='number of decoder layers'
)

parser.add_argument(
    '--d_ff',
    type=int,
    default=2048,
    help='dimension of feed-forward network'
)

parser.add_argument(
    '--moving_avg',
    type=int,
    default=25,
    help='window size of moving average'
)

parser.add_argument(
    '--factor',
    type=int,
    default=1,
    help='attention factor'
)

parser.add_argument(
    '--distil',
    action='store_false',
    default=True,
    help='whether to use distilling'
)

parser.add_argument(
    '--dropout',
    type=float,
    default=0,
    help='dropout'
)

parser.add_argument(
    '--embed',
    type=str,
    default='timeF',
    help='time features encoding'
)

parser.add_argument(
    '--activation',
    type=str,
    default='gelu',
    help='activation'
)

parser.add_argument(
    '--output_attention',
    action='store_true',
    help='whether to output attention'
)

parser.add_argument(
    '--do_predict',
    action='store_true',
    help='whether to predict unseen future data'
)


# ============================================================
# OPTIMIZATION
# ============================================================

parser.add_argument(
    '--num_workers',
    type=int,
    default=10,
    help='data loader workers'
)

parser.add_argument(
    '--itr',
    type=int,
    default=1,
    help='experiment repetitions'
)

parser.add_argument(
    '--train_epochs',
    type=int,
    default=30,
    help='training epochs'
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='batch size'
)

parser.add_argument(
    '--patience',
    type=int,
    default=5,
    help='early stopping patience'
)

parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.0001,
    help='optimizer learning rate'
)

parser.add_argument(
    '--des',
    type=str,
    default='test',
    help='experiment description'
)

parser.add_argument(
    '--loss',
    type=str,
    default='mse',
    help='loss function'
)

parser.add_argument(
    '--lradj',
    type=str,
    default='type3',
    help='learning-rate adjustment'
)

parser.add_argument(
    '--pct_start',
    type=float,
    default=0.3,
    help='pct_start'
)

parser.add_argument(
    '--use_amp',
    action='store_true',
    default=False,
    help='use automatic mixed precision'
)


# ============================================================
# GPU
# ============================================================

parser.add_argument(
    '--use_gpu',
    type=bool,
    default=True,
    help='use GPU'
)

parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help='GPU'
)

parser.add_argument(
    '--use_multi_gpu',
    action='store_true',
    default=False,
    help='use multiple GPUs'
)

parser.add_argument(
    '--devices',
    type=str,
    default='0,1',
    help='device IDs'
)

parser.add_argument(
    '--test_flop',
    action='store_true',
    default=False,
    help='test FLOPs'
)


# ============================================================
# PARSE ARGUMENTS
# ============================================================

args = parser.parse_args()


# ============================================================
# RANDOM SEED
# ============================================================

fix_seed = args.random_seed

random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


# ============================================================
# GPU SETUP
# ============================================================

args.use_gpu = (
    torch.cuda.is_available()
    and args.use_gpu
)

if args.use_gpu and args.use_multi_gpu:

    args.devices = args.devices.replace(' ', '')

    device_ids = args.devices.split(',')

    args.device_ids = [
        int(device_id)
        for device_id in device_ids
    ]

    args.gpu = args.device_ids[0]


print('Args in experiment:')
print(args)


# ============================================================
# EXPERIMENT
# ============================================================

Exp = Exp_Main


if args.is_training:

    for ii in range(args.itr):

        setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.cycle,
            args.model_type,
            fix_seed
        )

        exp = Exp(args)

        print(
            '>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'
            .format(setting)
        )

        exp.train(setting)

        print(
            '>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
            .format(setting)
        )

        exp.test(setting)

        if args.do_predict:

            print(
                '>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
                .format(setting)
            )

            exp.predict(
                setting,
                True
            )

        torch.cuda.empty_cache()


else:

    setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_{}_seed{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.cycle,
        args.model_type,
        fix_seed
    )

    exp = Exp(args)

    print(
        '>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
        .format(setting)
    )

    exp.test(
        setting,
        test=1
    )

    torch.cuda.empty_cache()