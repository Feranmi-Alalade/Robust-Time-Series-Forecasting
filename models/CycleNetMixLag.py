import torch
import torch.nn as nn


class RecurrentCycle(torch.nn.Module):
    # Identical to the original CycleNet RecurrentCycle (index-arithmetic gather).
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class LaggedChannelMixing(nn.Module):
    """
    Lagged generalization of the contemporaneous ChannelMixing block used in
    CycleNetMix.

    WHY THIS EXISTS: CycleNetMix's original mixing block only lets channels
    influence each other AT THE SAME TIMESTEP. But the cross-variable
    dependency evidence that motivated the mixing branch in the first place
    (Granger causality, Transfer Entropy) specifically measures LAGGED
    relationships -- whether channel X's PAST values inform channel Y's
    PRESENT/FUTURE value. A same-timestep-only block structurally cannot
    represent "congestion at sensor A now predicts congestion at sensor B
    fifteen minutes later" -- it can only ever compare simultaneous
    readings. Trained CycleNetMix checkpoints on Traffic showed a small but
    consistent nonzero gate (around -0.09 at both horizons) alongside a
    measurable INCREASE in test error, which is consistent with the
    contemporaneous block latching onto a spurious same-timestep pattern
    rather than the genuine lagged structure the dependency metrics found.

    HOW IT WORKS: implemented as a causal 1-D convolution across the
    channel axis with a kernel that spans `lag_window` timesteps. For each
    output channel and each timestep t, the convolution computes a weighted
    combination of every input channel's values at timesteps
    t, t-1, ..., t-(lag_window-1) -- i.e. it mixes channels AND lags
    simultaneously, rather than mixing channels first and lags separately.
    Left (past-side) zero-padding is used so the output at each timestep
    only ever looks backward in time, matching the "does X's past inform
    Y's present" semantics of Granger causality and Transfer Entropy. With
    lag_window=1 this collapses exactly to the original contemporaneous
    ChannelMixing block (a plain per-timestep channel mix), so the lagged
    version is a strict generalization, not a different mechanism.

    Same gating discipline as CycleNetMix: the branch's output is scaled by
    a learnable gate initialized to zero, so the model starts out identical
    to base CycleNet and only opens the lagged-mixing pathway if doing so
    reduces the training loss.
    """
    def __init__(self, channel_size, hidden_size, lag_window, dropout=0.1):
        super(LaggedChannelMixing, self).__init__()
        self.lag_window = lag_window
        self.norm = nn.LayerNorm(channel_size)
        # causal left-padding: pad (lag_window - 1) zeros before the sequence
        # so conv output length matches input length while only ever using
        # the current and past (lag_window - 1) timesteps.
        self.pad = nn.ConstantPad1d((lag_window - 1, 0), 0.0)
        self.mix_and_lag = nn.Conv1d(
            in_channels=channel_size, out_channels=hidden_size, kernel_size=lag_window
        )
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.project = nn.Conv1d(in_channels=hidden_size, out_channels=channel_size, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, seq_len, channels)
        residual = x
        h = self.norm(x)
        h = h.permute(0, 2, 1)              # (batch, channels, seq_len)
        h = self.pad(h)                     # (batch, channels, seq_len + lag_window - 1)
        h = self.mix_and_lag(h)             # (batch, hidden, seq_len)  -- mixes channels + lags jointly
        h = self.activation(h)
        h = self.dropout1(h)
        h = self.project(h)                 # (batch, channels, seq_len)
        h = self.dropout2(h)
        h = h.permute(0, 2, 1)              # (batch, seq_len, channels)
        return residual + self.gate * h


class Model(nn.Module):
    """
    CycleNet + gated, LAGGED channel-mixing branch.

    Identical pipeline to CycleNetMix except the mixing block (step 3) can
    see each channel's recent history, not just the current timestep:
        1. optional instance norm (RevIN-style)          -- unchanged
        2. remove learned cycle component                -- unchanged
        3. [MIX+LAG] exchange information across channels AND across a
           short lag window, on the residual
        4. channel-independent temporal backbone         -- unchanged
        5. add back cycle at future positions             -- unchanged
        6. optional instance denorm                       -- unchanged

    Extra config attributes (all optional, with safe defaults):
        mix_hidden     : hidden width of the mixing conv (default 128)
        mix_layers     : number of stacked mixing blocks (default 1)
        mix_dropout    : dropout inside the mixing block (default 0.1)
        mix_lag_window : number of past timesteps (including the current
                          one) each output position can see (default 4).
                          mix_lag_window=1 reproduces plain contemporaneous
                          mixing (equivalent to CycleNetMix).
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        mix_hidden = getattr(configs, 'mix_hidden', 128)
        mix_layers = getattr(configs, 'mix_layers', 1)
        mix_dropout = getattr(configs, 'mix_dropout', 0.1)
        mix_lag_window = getattr(configs, 'mix_lag_window', 4)

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        self.channel_mixing = nn.ModuleList([
            LaggedChannelMixing(self.enc_in, mix_hidden, mix_lag_window, mix_dropout)
            for _ in range(mix_layers)
        ])

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data
        x = x - self.cycleQueue(cycle_index, self.seq_len)

        # [MIX+LAG] cross-channel, cross-lag information exchange on the residual
        for block in self.channel_mixing:
            x = block(x)

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y