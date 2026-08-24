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
    """
    Moving average block to extract the trend of a series. Same padding
    convention as DLinear's series_decomp: replicate the first/last value
    to keep the output the same length as the input, so the trend and
    residual can be recombined directly.
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class TrendBranch(nn.Module):
    """
    Gated trend-extrapolation branch.

    WHY THIS EXISTS: CycleNet's backbone maps the cycle-removed residual
    straight to the future window with a single linear/MLP head, which has
    to absorb any remaining trend implicitly. On a dataset with strong
    trend (e.g. ETTm1, trend strength 0.734 from the STL characterization),
    that is a lot to ask of one shared mapping -- and it's exactly the
    failure mode the seasonal-naive analysis surfaced earlier: a forecast
    that can't represent drift is systematically wrong in whichever
    direction the series has moved, with the error compounding over the
    horizon.

    HOW IT WORKS: extracts a smoothed trend component from the
    cycle-removed input via a moving average (same technique as DLinear),
    then extrapolates that trend forward with its own linear map
    (seq_len -> pred_len), separate from the main backbone. This gives the
    model an explicit pathway for "the series has been drifting, continue
    that drift" that doesn't have to be learned implicitly inside the same
    weights that also handle cycle-free short-term variation.

    Same gating discipline as the mixing branches: the branch's output is
    scaled by a learnable gate initialized to zero, so the model starts
    out identical to base CycleNet (whose own backbone runs on the full,
    non-decomposed residual, unchanged) and only opens the trend pathway
    if doing so reduces the training loss. This also means the two designs
    are directly comparable in the same ablation: CycleNet, CycleNetMix
    (channel mixing), and CycleNetTrend (trend extrapolation) all reduce
    to the identical base model at gate=0.
    """
    def __init__(self, seq_len, pred_len, moving_avg_kernel):
        super(TrendBranch, self).__init__()
        self.decompose = MovingAvg(moving_avg_kernel, stride=1)
        self.trend_head = nn.Linear(seq_len, pred_len)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, seq_len, channels) -- the cycle-removed residual
        trend = self.decompose(x)                          # (batch, seq_len, channels)
        trend = trend.permute(0, 2, 1)                      # (batch, channels, seq_len)
        trend_forecast = self.trend_head(trend)              # (batch, channels, pred_len)
        trend_forecast = trend_forecast.permute(0, 2, 1)     # (batch, pred_len, channels)
        return self.gate * trend_forecast


class Model(nn.Module):
    """
    CycleNet + gated trend-extrapolation branch.

    Pipeline (differences from base CycleNet marked with [TREND]):
        1. optional instance norm (RevIN-style)          -- unchanged
        2. remove learned cycle component                -- unchanged
        3. channel-independent temporal backbone, run on the FULL
           cycle-removed residual                          -- unchanged
        4. [TREND] separately, extract + extrapolate a smoothed trend
           component from the same residual, and add it in (gated)
        5. add back cycle at future positions              -- unchanged
        6. optional instance denorm                        -- unchanged

    Extra config attributes (optional, safe default):
        trend_kernel : moving-average window used to extract the trend
                       component (default: configs.moving_avg if present,
                       else 25 -- matching DLinear's default). Should be
                       odd, and roughly on the scale of the dataset's
                       short-term noise you want smoothed away (not the
                       seasonal period -- that's already handled by the
                       cycle component).
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

        trend_kernel = getattr(configs, 'trend_kernel', None) or getattr(configs, 'moving_avg', 25)
        if trend_kernel % 2 == 0:
            trend_kernel += 1  # MovingAvg's padding assumes an odd kernel

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
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

        # main backbone, unchanged -- runs on the full cycle-removed residual
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # [TREND] gated trend-extrapolation correction, added on top
        y = y + self.trend_branch(x)

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y