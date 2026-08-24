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


class ChannelMixing(nn.Module):
    """
    TSMixer-style channel-mixing block applied to the cycle-removed residual.

    At every time step, information is exchanged across channels via a
    two-layer MLP over the channel dimension, wrapped in LayerNorm and a
    residual connection. A learnable scalar gate, initialized to ZERO,
    scales the mixing contribution:

        out = x + gate * MLP(LayerNorm(x))

    Zero-initialization means the model starts out *exactly* equivalent to
    vanilla CycleNet (the branch contributes nothing at step 0), and only
    grows the mixing pathway if it reduces the loss. This matters for the
    experimental design: on a dataset with weak cross-variable dependency
    (e.g. ETTm1) the gate can stay near zero and performance should match
    base CycleNet, while on a dataset with strong cross-variable dependency
    (e.g. Traffic) the gate can open. The learned |gate| value is itself an
    interpretable readout of how much cross-channel information the model
    found useful.
    """
    def __init__(self, channel_size, hidden_size, dropout=0.1):
        super(ChannelMixing, self).__init__()
        self.norm = nn.LayerNorm(channel_size)
        self.mlp = nn.Sequential(
            nn.Linear(channel_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, channel_size),
            nn.Dropout(dropout),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, seq_len, channels) -- mixing acts on the last (channel) dim
        return x + self.gate * self.mlp(self.norm(x))


class Model(nn.Module):
    """
    CycleNet + gated channel-mixing branch.

    Pipeline (differences from base CycleNet marked with [MIX]):
        1. optional instance norm (RevIN-style)          -- unchanged
        2. remove learned cycle component                -- unchanged
        3. [MIX] exchange information across channels on the residual
        4. channel-independent temporal backbone         -- unchanged
        5. add back cycle at future positions            -- unchanged
        6. optional instance denorm                      -- unchanged

    The mixing is applied to the *cycle-removed residual* (step 3), not the
    raw input: the shared periodic component is already captured per-channel
    by the cycle queue, so what remains is exactly the signal in which
    genuine cross-channel information (e.g. congestion propagating between
    road sensors) would live.

    Extra config attributes (with safe defaults if absent):
        mix_hidden : hidden width of the channel-mixing MLP (default 128)
        mix_layers : number of stacked mixing blocks (default 1)
        mix_dropout: dropout inside the mixing MLP (default 0.1)
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

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        # [MIX] gated channel-mixing block(s) on the cycle-removed residual
        self.channel_mixing = nn.Sequential(*[
            ChannelMixing(self.enc_in, mix_hidden, mix_dropout)
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

        # [MIX] cross-channel information exchange on the residual
        x = self.channel_mixing(x)

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y