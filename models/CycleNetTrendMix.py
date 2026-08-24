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


class MovingAvg(nn.Module):
    """Same padding convention as DLinear's series_decomp (see CycleNetTrend)."""
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class TrendBranch(nn.Module):
    """Identical to CycleNetTrend.TrendBranch -- see that file for the full
    rationale. Own gate, initialized to zero."""
    def __init__(self, seq_len, pred_len, moving_avg_kernel):
        super(TrendBranch, self).__init__()
        self.decompose = MovingAvg(moving_avg_kernel, stride=1)
        self.trend_head = nn.Linear(seq_len, pred_len)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        trend = self.decompose(x)
        trend = trend.permute(0, 2, 1)
        trend_forecast = self.trend_head(trend)
        trend_forecast = trend_forecast.permute(0, 2, 1)
        return self.gate * trend_forecast


class LowRankChannelMixing(nn.Module):
    """Identical to CycleNetMixLowRank.LowRankChannelMixing -- see that file
    for the full rationale. Own gate, initialized to zero."""
    def __init__(self, channel_size, rank, dropout=0.1):
        super(LowRankChannelMixing, self).__init__()
        self.norm = nn.LayerNorm(channel_size)
        self.encode = nn.Linear(channel_size, rank, bias=False)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.decode = nn.Linear(rank, channel_size, bias=True)
        self.dropout2 = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = self.encode(h)
        h = self.activation(h)
        h = self.dropout1(h)
        h = self.decode(h)
        h = self.dropout2(h)
        return residual + self.gate * h


class Model(nn.Module):
    """
    CycleNet + gated trend-extrapolation branch + gated low-rank
    channel-mixing branch, combined in one model.

    Pipeline (differences from base CycleNet marked with [TREND] / [MIX]):
        1. optional instance norm (RevIN-style)              -- unchanged
        2. remove learned cycle component                    -- unchanged
        3. [MIX] channel mixing on the cycle-removed residual, gated,
           feeds into the backbone (so a genuinely mixed representation
           reaches the temporal model, not just an ignorable side-input)
        4. channel-independent temporal backbone on the (possibly mixed)
           residual                                             -- unchanged mechanism
        5. [TREND] separately, extract + extrapolate a trend component
           from the ORIGINAL (pre-mix) cycle-removed residual, gated,
           and add it in on top of the backbone's output
        6. add back cycle at future positions                  -- unchanged
        7. optional instance denorm                             -- unchanged

    Both gates are independent and both initialized to zero, so the model
    starts out identical to base CycleNet regardless of which branch (if
    any) is present. This has a useful side effect for interpretation: you
    do not need to run four separate architectures (neither / mix only /
    trend only / both) to test whether combining the two helps -- you can
    train THIS model once per dataset and read off both learned gate
    values afterward. If a dataset has both strong trend and strong
    cross-variable dependency, both gates opening would directly confirm
    the two effects combine; if only one opens, that tells you the other
    property wasn't actually usable once its counterpart was available
    too (e.g. the mixing branch's contribution might change once the
    backbone no longer has to work as hard on trend, or vice versa).

    Why the mixing branch feeds into the backbone here (unlike the
    standalone CycleNetMixLowRank, where it does the same thing) but the
    trend branch is additive on top of the backbone's output (unlike a
    strict three-way decomposition): this mirrors each branch's own
    standalone design exactly, so results from the standalone ablations
    remain the closest possible reference point for interpreting this
    combined model's behavior.

    Extra config attributes (all optional, with safe defaults):
        mix_rank      : low-rank dimension for channel mixing (default 16)
        mix_layers    : number of stacked low-rank mixing blocks (default 1)
        mix_dropout   : dropout inside the mixing block (default 0.1)
        trend_kernel  : moving-average window for the trend branch
                        (default: configs.moving_avg if present, else 25)
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

        mix_rank = getattr(configs, 'mix_rank', 16)
        mix_layers = getattr(configs, 'mix_layers', 1)
        mix_dropout = getattr(configs, 'mix_dropout', 0.1)
        trend_kernel = getattr(configs, 'trend_kernel', None) or getattr(configs, 'moving_avg', 25)
        if trend_kernel % 2 == 0:
            trend_kernel += 1

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        self.channel_mixing = nn.ModuleList([
            LowRankChannelMixing(self.enc_in, mix_rank, mix_dropout)
            for _ in range(mix_layers)
        ])
        self.trend_branch = TrendBranch(self.seq_len, self.pred_len, trend_kernel)

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

        # [TREND] extrapolate trend from the original cycle-removed residual
        trend_correction = self.trend_branch(x)

        # [MIX] channel mixing feeds into what the backbone actually sees
        x_mixed = x
        for block in self.channel_mixing:
            x_mixed = block(x_mixed)

        # main backbone, on the (possibly mixed) residual
        y = self.model(x_mixed.permute(0, 2, 1)).permute(0, 2, 1)

        # add the gated trend correction on top
        y = y + trend_correction

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y