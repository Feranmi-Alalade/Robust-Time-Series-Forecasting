import torch
import torch.nn as nn


# ============================================================
# RECURRENT CYCLE
# ============================================================

class RecurrentCycle(nn.Module):
    """
    Original CycleNet recurrent cycle.

    Learnable cycle matrix:

        Q ∈ R^(cycle_len × channel_size)

    Each row corresponds to one position in the cycle.
    Each column corresponds to one channel.
    """

    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()

        self.cycle_len = cycle_len
        self.channel_size = channel_size

        self.data = nn.Parameter(
            torch.zeros(cycle_len, channel_size),
            requires_grad=True
        )

    def forward(self, index, length):
        """
        Parameters
        ----------
        index:
            Cycle position of the first timestep
            of each input window.

            Shape:
                (batch_size,)

        length:
            Number of consecutive cycle positions required.

        Returns
        -------
        Cycle values with shape:

            (batch_size, length, channel_size)
        """

        gather_index = (
            index.view(-1, 1)
            +
            torch.arange(
                length,
                device=index.device
            ).view(1, -1)
        ) % self.cycle_len

        return self.data[gather_index]


# ============================================================
# CHANNEL MIXING
# ============================================================

class ChannelMixing(nn.Module):
    """
    TSMixer-style channel-mixing block applied to the
    cycle-removed residual.

    Input shape:

        (batch, seq_len, channels)

    At every timestep, the complete channel vector:

        [x_1, x_2, ..., x_D]

    is processed by an MLP.

    This allows information from one channel to influence
    the representation of another channel.

    Equation:

        out = x + gate * MLP(LayerNorm(x))

    The learnable gate is initialized to zero.

    Therefore, at initialization:

        out = x

    and the additional mixing branch initially contributes
    nothing.
    """

    def __init__(
        self,
        channel_size,
        hidden_size,
        dropout=0.1
    ):
        super(ChannelMixing, self).__init__()

        # Normalize across channels
        self.norm = nn.LayerNorm(channel_size)

        # Channel mixing MLP:
        #
        # channels -> hidden -> channels
        self.mlp = nn.Sequential(
            nn.Linear(
                channel_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_size,
                channel_size
            ),

            nn.Dropout(dropout)
        )

        # Learnable scalar gate
        #
        # Starts at zero so the branch initially
        # contributes nothing.
        self.gate = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, x):
        """
        x shape:

            (batch, seq_len, channels)
        """

        mixed = self.mlp(
            self.norm(x)
        )

        return (
            x
            +
            self.gate * mixed
        )


# ============================================================
# CYCLETNET MIX TUNING MODEL
# ============================================================

class Model(nn.Module):
    """
    CycleNet + gated channel-mixing branch.

    This tuning version keeps the channel-mixing width and
    dropout FIXED and tunes only:

        1. mix_layers
        2. learning_rate

    learning_rate is NOT handled inside this model.
    It is supplied through run.py and used by Exp_Main's
    optimizer.

    ----------------------------------------------------------
    FIXED MIXER SETTINGS
    ----------------------------------------------------------

        mix_hidden  = 64
        mix_dropout = 0.1

    ----------------------------------------------------------
    TUNED SETTING
    ----------------------------------------------------------

        mix_layers

    Suggested values:

        1
        2

    Learning-rate values are supplied externally, e.g.:

        1e-4
        5e-4
        1e-3


    ==========================================================
    PIPELINE
    ==========================================================

    Input:

        X ∈ R^(B × L × D)

    1. Optional instance normalization

        X -> X_norm

    2. Remove learned cycle

        R = X_norm - C_past

    3. Cross-channel mixing

        R_mix =
            R
            +
            gate * MLP(LayerNorm(R))

    4. Original CycleNet temporal backbone

        R_mix
            -> Linear / MLP
            -> future residual

    5. Add future cycle

        Y_norm =
            residual_forecast
            +
            C_future

    6. Optional instance denormalization
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # ====================================================
        # ORIGINAL CYCLETNET PARAMETERS
        # ====================================================

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.enc_in = configs.enc_in

        self.cycle_len = configs.cycle

        self.model_type = configs.model_type

        self.d_model = configs.d_model

        self.use_revin = configs.use_revin


        # ====================================================
        # CHANNEL MIXER SETTINGS
        # ====================================================

        # Fixed to reduce hyperparameter-search cost.
        self.mix_hidden = 64

        # Fixed regularization.
        self.mix_dropout = 0.1

        # ONLY mixer architecture parameter being tuned.
        self.mix_layers = getattr(
            configs,
            "mix_layers",
            1
        )


        # ====================================================
        # LEARNABLE RECURRENT CYCLE
        # ====================================================

        self.cycleQueue = RecurrentCycle(
            cycle_len=self.cycle_len,
            channel_size=self.enc_in
        )


        # ====================================================
        # CHANNEL MIXING BLOCK(S)
        # ====================================================

        self.channel_mixing = nn.Sequential(
            *[
                ChannelMixing(
                    channel_size=self.enc_in,
                    hidden_size=self.mix_hidden,
                    dropout=self.mix_dropout
                )
                for _ in range(self.mix_layers)
            ]
        )


        # ====================================================
        # ORIGINAL CYCLETNET TEMPORAL BACKBONE
        # ====================================================

        assert self.model_type in [
            "linear",
            "mlp"
        ]

        if self.model_type == "linear":

            self.model = nn.Linear(
                self.seq_len,
                self.pred_len
            )

        elif self.model_type == "mlp":

            self.model = nn.Sequential(

                nn.Linear(
                    self.seq_len,
                    self.d_model
                ),

                nn.ReLU(),

                nn.Linear(
                    self.d_model,
                    self.pred_len
                )
            )


    def forward(self, x, cycle_index):
        """
        Parameters
        ----------
        x:
            Input time series.

            Shape:
                (batch_size, seq_len, enc_in)

        cycle_index:
            Cycle position of the FIRST timestep
            in every lookback window.

            Shape:
                (batch_size,)
        """


        # ====================================================
        # 1. OPTIONAL INSTANCE NORMALIZATION
        # ====================================================

        if self.use_revin:

            seq_mean = torch.mean(
                x,
                dim=1,
                keepdim=True
            )

            seq_var = (
                torch.var(
                    x,
                    dim=1,
                    keepdim=True
                )
                + 1e-5
            )

            x = (
                x - seq_mean
            ) / torch.sqrt(seq_var)


        # ====================================================
        # 2. REMOVE LEARNED CYCLE
        # ====================================================

        past_cycle = self.cycleQueue(
            cycle_index,
            self.seq_len
        )

        x = x - past_cycle


        # At this point:
        #
        # x = cycle-removed residual
        #
        # R = X - C_past
        #
        # Shape:
        #
        # (batch, seq_len, channels)


        # ====================================================
        # 3. CROSS-CHANNEL MIXING
        # ====================================================

        x = self.channel_mixing(x)


        # ====================================================
        # 4. TEMPORAL FORECASTING
        # ====================================================
        #
        # Current:
        #
        #     (B, L, D)
        #
        # Change to:
        #
        #     (B, D, L)
        #
        # because the CycleNet backbone maps:
        #
        #     seq_len -> pred_len
        #

        x_temporal = x.permute(
            0,
            2,
            1
        )


        y = self.model(
            x_temporal
        )


        # Backbone output:
        #
        #     (B, D, H)
        #
        # Restore:
        #
        #     (B, H, D)

        y = y.permute(
            0,
            2,
            1
        )


        # ====================================================
        # 5. ADD FUTURE CYCLE
        # ====================================================

        future_cycle_index = (
            cycle_index
            +
            self.seq_len
        ) % self.cycle_len


        future_cycle = self.cycleQueue(
            future_cycle_index,
            self.pred_len
        )


        y = y + future_cycle


        # ====================================================
        # 6. OPTIONAL INSTANCE DENORMALIZATION
        # ====================================================

        if self.use_revin:

            y = (
                y
                * torch.sqrt(seq_var)
                + seq_mean
            )


        return y