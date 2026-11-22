from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, \
    LDLinear, SparseTSF, RLinear, RMLP, CycleiTransformer, CycleNetMix, CycleNetMixTune, CycleNetMixLag, \
    CycleNetMixLowRank, CycleNetPairMix, CycleNetTrend, CycleNetTrendMultiScale, CycleNetTrendDecomp, CycleNetTrendMix, CycleNetTrendMixDense

from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import warnings

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    # ============================================================
    # MODEL
    # ============================================================

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SegRNN': SegRNN,
            'CycleNet': CycleNet,
            'LDLinear': LDLinear,
            'SparseTSF': SparseTSF,
            'RLinear': RLinear,
            'RMLP': RMLP,
            'CycleiTransformer': CycleiTransformer,
            'CycleNetMix': CycleNetMix,
            'CycleNetMixTune': CycleNetMixTune,
            'CycleNetMixLag': CycleNetMixLag,
            'CycleNetMixLowRank': CycleNetMixLowRank,
            'CycleNetPairMix': CycleNetPairMix,
            'CycleNetTrend': CycleNetTrend,
            'CycleNetTrendMultiScale': CycleNetTrendMultiScale,
            'CycleNetTrendDecomp': CycleNetTrendDecomp,
            'CycleNetTrendMix': CycleNetTrendMix,
            'CycleNetTrendMixDense': CycleNetTrendMixDense
        }

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    # ============================================================
    # DATA
    # ============================================================

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # ============================================================
    # OPTIMIZER / LOSS
    # ============================================================

    def _select_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate
        )

    def _select_criterion(self):
        return nn.MSELoss()

    # ============================================================
    # SHARED FORWARD ROUTING
    # ============================================================

    def _forward_model(
        self,
        batch_x,
        batch_x_mark,
        dec_inp,
        batch_y_mark,
        batch_cycle
    ):

        if 'Cycle' in self.args.model:
            outputs = self.model(batch_x, batch_cycle)

        elif any(
            substr in self.args.model
            for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}
        ):
            outputs = self.model(batch_x)

        else:
            outputs = self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark
            )

            if self.args.output_attention:
                outputs = outputs[0]

        return outputs

    # ============================================================
    # VALIDATION
    #
    # IMPORTANT:
    # Returns BOTH validation MSE and validation MAE.
    #
    # Validation MSE is used:
    #   1. for early stopping
    #   2. for Optuna hyperparameter selection
    #
    # Validation MAE is recorded for the SAME checkpoint.
    # ============================================================

    def vali(self, vali_data, vali_loader, criterion):

        mse_losses = []
        mae_losses = []

        self.model.eval()

        with torch.no_grad():

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_cycle
            ) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # ------------------------------------------------
                # Decoder input
                # ------------------------------------------------

                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]
                ).float()

                dec_inp = torch.cat(
                    [
                        batch_y[:, :self.args.label_len, :],
                        dec_inp
                    ],
                    dim=1
                ).float().to(self.device)

                # ------------------------------------------------
                # Forward
                # ------------------------------------------------

                if self.args.use_amp:

                    with torch.cuda.amp.autocast():

                        outputs = self._forward_model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                            batch_cycle
                        )

                else:

                    outputs = self._forward_model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                        batch_cycle
                    )

                # ------------------------------------------------
                # Target selection
                # ------------------------------------------------

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[
                    :,
                    -self.args.pred_len:,
                    f_dim:
                ]

                batch_y = batch_y[
                    :,
                    -self.args.pred_len:,
                    f_dim:
                ].to(self.device)

                # ------------------------------------------------
                # Validation metrics
                # ------------------------------------------------

                mse = criterion(outputs, batch_y)
                mae = torch.mean(torch.abs(outputs - batch_y))

                mse_losses.append(mse.item())
                mae_losses.append(mae.item())

        val_mse = float(np.average(mse_losses))
        val_mae = float(np.average(mae_losses))

        self.model.train()

        return val_mse, val_mae

    # ============================================================
    # TRAIN
    #
    # IMPORTANT CHANGE:
    #
    # Test data is NOT evaluated during training.
    #
    # The best model is selected ONLY using validation MSE.
    #
    # train() returns:
    #
    # {
    #     'val_mse': ...,
    #     'val_mae': ...,
    #     'best_epoch': ...
    # }
    #
    # Optuna should minimize result['val_mse'].
    # ============================================================

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(
            self.args.checkpoints,
            setting
        )

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # --------------------------------------------------------
        # Keep track of best validation checkpoint
        # --------------------------------------------------------

        best_val_mse = float('inf')
        best_val_mae = float('inf')
        best_epoch = -1

        # --------------------------------------------------------
        # AMP
        # --------------------------------------------------------

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # --------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        # ========================================================
        # EPOCH LOOP
        # ========================================================

        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()

            # ====================================================
            # TRAINING LOOP
            # ====================================================

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_cycle
            ) in enumerate(train_loader):

                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # ------------------------------------------------
                # Decoder input
                # ------------------------------------------------

                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]
                ).float()

                dec_inp = torch.cat(
                    [
                        batch_y[:, :self.args.label_len, :],
                        dec_inp
                    ],
                    dim=1
                ).float().to(self.device)

                # ------------------------------------------------
                # Forward
                # ------------------------------------------------

                if self.args.use_amp:

                    with torch.cuda.amp.autocast():

                        outputs = self._forward_model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                            batch_cycle
                        )

                        f_dim = -1 if self.args.features == 'MS' else 0

                        outputs = outputs[
                            :,
                            -self.args.pred_len:,
                            f_dim:
                        ]

                        target = batch_y[
                            :,
                            -self.args.pred_len:,
                            f_dim:
                        ]

                        loss = criterion(
                            outputs,
                            target
                        )

                else:

                    outputs = self._forward_model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                        batch_cycle
                    )

                    f_dim = -1 if self.args.features == 'MS' else 0

                    outputs = outputs[
                        :,
                        -self.args.pred_len:,
                        f_dim:
                    ]

                    target = batch_y[
                        :,
                        -self.args.pred_len:,
                        f_dim:
                    ]

                    loss = criterion(
                        outputs,
                        target
                    )

                train_loss.append(
                    loss.item()
                )

                # ------------------------------------------------
                # Logging
                # ------------------------------------------------

                if (i + 1) % 100 == 0:

                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1,
                            epoch + 1,
                            loss.item()
                        )
                    )

                    speed = (
                        time.time() - time_now
                    ) / iter_count

                    left_time = speed * (
                        (
                            self.args.train_epochs - epoch
                        ) * train_steps - i
                    )

                    print(
                        '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(
                            speed,
                            left_time
                        )
                    )

                    iter_count = 0
                    time_now = time.time()

                # ------------------------------------------------
                # Backpropagation
                # ------------------------------------------------

                if self.args.use_amp:

                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()

                else:

                    loss.backward()
                    model_optim.step()

                # ------------------------------------------------
                # TST learning-rate scheduler
                # ------------------------------------------------

                if self.args.lradj == 'TST':

                    adjust_learning_rate(
                        model_optim,
                        scheduler,
                        epoch + 1,
                        self.args,
                        printout=False
                    )

                    scheduler.step()

            # ====================================================
            # END OF EPOCH
            # ====================================================

            print(
                "Epoch: {} cost time: {}".format(
                    epoch + 1,
                    time.time() - epoch_time
                )
            )

            train_loss = float(
                np.average(train_loss)
            )

            # ----------------------------------------------------
            # VALIDATION ONLY
            #
            # No test-set evaluation here.
            # ----------------------------------------------------

            val_mse, val_mae = self.vali(
                vali_data,
                vali_loader,
                criterion
            )

            print(
                "Epoch: {0}, Steps: {1} | "
                "Train Loss: {2:.7f} "
                "Val MSE: {3:.7f} "
                "Val MAE: {4:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    val_mse,
                    val_mae
                )
            )

            # ----------------------------------------------------
            # Track best validation checkpoint
            # ----------------------------------------------------

            if val_mse < best_val_mse:

                best_val_mse = val_mse
                best_val_mae = val_mae
                best_epoch = epoch + 1

            # ----------------------------------------------------
            # Early stopping based ONLY on validation MSE
            # ----------------------------------------------------

            early_stopping(
                val_mse,
                self.model,
                path
            )

            if early_stopping.early_stop:

                print("Early stopping")

                break

            # ----------------------------------------------------
            # Learning-rate update
            # ----------------------------------------------------

            if self.args.lradj != 'TST':

                adjust_learning_rate(
                    model_optim,
                    scheduler,
                    epoch + 1,
                    self.args
                )

            else:

                print(
                    'Updating learning rate to {}'.format(
                        scheduler.get_last_lr()[0]
                    )
                )

        # ========================================================
        # LOAD BEST VALIDATION CHECKPOINT
        # ========================================================

        best_model_path = os.path.join(
            path,
            'checkpoint.pth'
        )

        self.model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=self.device
            )
        )

        # --------------------------------------------------------
        # Re-evaluate the SAVED best checkpoint on validation data
        #
        # This ensures the returned validation MSE/MAE correspond
        # exactly to the checkpoint that will later be tested.
        # --------------------------------------------------------

        final_val_mse, final_val_mae = self.vali(
            vali_data,
            vali_loader,
            criterion
        )

        print("")
        print("======================================================")
        print("BEST VALIDATION CHECKPOINT")
        print("======================================================")
        print("Best epoch:          {}".format(best_epoch))
        print("Validation MSE:      {:.8f}".format(final_val_mse))
        print("Validation MAE:      {:.8f}".format(final_val_mae))
        print("======================================================")
        print("")

        return {
            'val_mse': final_val_mse,
            'val_mae': final_val_mae,
            'best_epoch': best_epoch
        }

    # ============================================================
    # TEST
    #
    # This should be called AFTER hyperparameter selection.
    #
    # Optuna must NOT use these values as its objective.
    # ============================================================

    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(
            flag='test'
        )

        if test:

            print('loading model')

            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        './checkpoints/',
                        setting,
                        'checkpoint.pth'
                    ),
                    map_location=self.device
                )
            )

        preds = []
        trues = []

        folder_path = os.path.join(
            './test_results/',
            setting
        )

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        with torch.no_grad():

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_cycle
            ) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # ------------------------------------------------
                # Decoder input
                # ------------------------------------------------

                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]
                ).float()

                dec_inp = torch.cat(
                    [
                        batch_y[:, :self.args.label_len, :],
                        dec_inp
                    ],
                    dim=1
                ).float().to(self.device)

                # ------------------------------------------------
                # Forward
                # ------------------------------------------------

                if self.args.use_amp:

                    with torch.cuda.amp.autocast():

                        outputs = self._forward_model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                            batch_cycle
                        )

                else:

                    outputs = self._forward_model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                        batch_cycle
                    )

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[
                    :,
                    -self.args.pred_len:,
                    f_dim:
                ]

                batch_y = batch_y[
                    :,
                    -self.args.pred_len:,
                    f_dim:
                ]

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                # ------------------------------------------------
                # Visualization
                # ------------------------------------------------

                if i % 20 == 0:

                    input_data = (
                        batch_x
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    gt = np.concatenate(
                        (
                            input_data[0, :, -1],
                            true[0, :, -1]
                        ),
                        axis=0
                    )

                    pd = np.concatenate(
                        (
                            input_data[0, :, -1],
                            pred[0, :, -1]
                        ),
                        axis=0
                    )

                    visual(
                        gt,
                        pd,
                        os.path.join(
                            folder_path,
                            str(i) + '.pdf'
                        )
                    )

                    np.savetxt(
                        os.path.join(
                            folder_path,
                            str(i) + '.txt'
                        ),
                        pd
                    )

                    np.savetxt(
                        os.path.join(
                            folder_path,
                            str(i) + 'true.txt'
                        ),
                        gt
                    )

        # --------------------------------------------------------
        # FLOPs
        # --------------------------------------------------------

        if self.args.test_flop:

            test_params_flop(
                self.model,
                (
                    batch_x.shape[1],
                    batch_x.shape[2]
                )
            )

            exit()

        # --------------------------------------------------------
        # Combine predictions
        # --------------------------------------------------------

        preds = np.concatenate(
            preds,
            axis=0
        )

        trues = np.concatenate(
            trues,
            axis=0
        )

        preds = preds.reshape(
            -1,
            preds.shape[-2],
            preds.shape[-1]
        )

        trues = trues.reshape(
            -1,
            trues.shape[-2],
            trues.shape[-1]
        )

        # --------------------------------------------------------
        # Results directory
        # --------------------------------------------------------

        result_path = os.path.join(
            './results/',
            setting
        )

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # --------------------------------------------------------
        # Test metrics
        # --------------------------------------------------------

        mae, mse, rmse, mape, mspe, rse, corr = metric(
            preds,
            trues
        )

        print("")
        print("======================================================")
        print("FINAL TEST RESULTS")
        print("======================================================")
        print("MSE: {:.8f}".format(mse))
        print("MAE: {:.8f}".format(mae))
        print("======================================================")
        print("")

        # --------------------------------------------------------
        # Save results
        # --------------------------------------------------------

        with open(
            "result.txt",
            'a'
        ) as f:

            f.write(
                setting + "\n"
            )

            f.write(
                'mse:{}, mae:{}\n\n'.format(
                    mse,
                    mae
                )
            )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Return metrics so final Optuna-selected configuration
        # can be evaluated programmatically.
        # --------------------------------------------------------

        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
            'rse': float(rse),
            'corr': float(np.asarray(corr).mean())
        }

    # ============================================================
    # PREDICT
    # ============================================================

    def predict(self, setting, load=False):

        pred_data, pred_loader = self._get_data(
            flag='pred'
        )

        if load:

            path = os.path.join(
                self.args.checkpoints,
                setting
            )

            best_model_path = os.path.join(
                path,
                'checkpoint.pth'
            )

            self.model.load_state_dict(
                torch.load(
                    best_model_path,
                    map_location=self.device
                )
            )

        preds = []

        self.model.eval()

        with torch.no_grad():

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_cycle
            ) in enumerate(pred_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # ------------------------------------------------
                # Decoder input
                # ------------------------------------------------

                dec_inp = torch.zeros(
                    [
                        batch_y.shape[0],
                        self.args.pred_len,
                        batch_y.shape[2]
                    ]
                ).float().to(
                    batch_y.device
                )

                dec_inp = torch.cat(
                    [
                        batch_y[:, :self.args.label_len, :],
                        dec_inp
                    ],
                    dim=1
                ).float().to(self.device)

                # ------------------------------------------------
                # Forward
                # ------------------------------------------------

                if self.args.use_amp:

                    with torch.cuda.amp.autocast():

                        outputs = self._forward_model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                            batch_cycle
                        )

                else:

                    outputs = self._forward_model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                        batch_cycle
                    )

                pred = (
                    outputs
                    .detach()
                    .cpu()
                    .numpy()
                )

                preds.append(
                    pred
                )

        preds = np.array(
            preds
        )

        preds = preds.reshape(
            -1,
            preds.shape[-2],
            preds.shape[-1]
        )

        folder_path = os.path.join(
            './results/',
            setting
        )

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(
            os.path.join(
                folder_path,
                'real_prediction.npy'
            ),
            preds
        )

        return