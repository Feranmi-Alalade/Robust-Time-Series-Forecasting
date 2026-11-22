import torch
import torch.nn as nn


class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()

        self.cycle_len = cycle_len
        self.channel_size = channel_size

        self.data = nn.Parameter(
            torch.zeros(cycle_len, channel_size),
            requires_grad=True
        )

    def forward(self, index, length):
        gather_index = (
            index.view(-1, 1)
            + torch.arange(
                length,
                device=index.device
            ).view(1, -1)
        ) % self.cycle_len

        return self.data[gather_index]


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()

        self.kernel_size = kernel_size

        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )

    def forward(self, x):
        front = x[:, 0:1, :].repeat(
            1,
            (self.kernel_size - 1) // 2,
            1
        )

        end = x[:, -1:, :].repeat(
            1,
            (self.kernel_size - 1) // 2,
            1
        )

        x = torch.cat(
            [front, x, end],
            dim=1
        )

        x = self.avg(
            x.permute(0, 2, 1)
        )

        return x.permute(0, 2, 1)


class MultiScaleTrendBranch(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        moving_avg_kernels,
        trend_layers,
        d_model
    ):
        super(MultiScaleTrendBranch, self).__init__()

        self.moving_avg_kernels = moving_avg_kernels

        self.decompose = nn.ModuleList([
            MovingAvg(kernel, stride=1)
            for kernel in moving_avg_kernels
        ])

        self.scale_logits = nn.Parameter(
            torch.zeros(len(moving_avg_kernels))
        )

        if trend_layers == 1:
            self.trend_head = nn.Linear(
                seq_len,
                pred_len
            )

        elif trend_layers == 2:
            self.trend_head = nn.Sequential(
                nn.Linear(
                    seq_len,
                    d_model
                ),
                nn.ReLU(),
                nn.Linear(
                    d_model,
                    pred_len
                )
            )

        else:
            raise ValueError(
                "trend_layers must be 1 or 2"
            )

        self.gate = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, x):
        trends = [
            moving_avg(x)
            for moving_avg in self.decompose
        ]

        trends = torch.stack(
            trends,
            dim=0
        )

        scale_weights = torch.softmax(
            self.scale_logits,
            dim=0
        )

        combined_trend = (
            scale_weights.view(-1, 1, 1, 1)
            * trends
        ).sum(dim=0)

        combined_trend = combined_trend.permute(
            0,
            2,
            1
        )

        trend_forecast = self.trend_head(
            combined_trend
        )

        trend_forecast = trend_forecast.permute(
            0,
            2,
            1
        )

        return self.gate * trend_forecast


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        self.trend_layers = getattr(
            configs,
            "trend_layers",
            1
        )

        self.trend_kernels = getattr(
            configs,
            "trend_kernels",
            [25, 49, 97]
        )

        self.trend_kernels = [
            kernel if kernel % 2 == 1 else kernel + 1
            for kernel in self.trend_kernels
        ]

        self.cycleQueue = RecurrentCycle(
            self.cycle_len,
            self.enc_in
        )

        self.trend_branch = MultiScaleTrendBranch(
            self.seq_len,
            self.pred_len,
            self.trend_kernels,
            self.trend_layers,
            self.d_model
        )

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
        if self.use_revin:
            seq_mean = torch.mean(
                x,
                dim=1,
                keepdim=True
            )

            seq_var = torch.var(
                x,
                dim=1,
                keepdim=True
            ) + 1e-5

            x = (
                x - seq_mean
            ) / torch.sqrt(seq_var)

        x = x - self.cycleQueue(
            cycle_index,
            self.seq_len
        )

        y = self.model(
            x.permute(0, 2, 1)
        ).permute(
            0,
            2,
            1
        )

        y = y + self.trend_branch(x)

        y = y + self.cycleQueue(
            (
                cycle_index
                + self.seq_len
            ) % self.cycle_len,
            self.pred_len
        )

        if self.use_revin:
            y = (
                y * torch.sqrt(seq_var)
                + seq_mean
            )

        return y