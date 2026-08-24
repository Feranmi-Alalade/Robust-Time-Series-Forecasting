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


class LowRankChannelMixing(nn.Module):
    """
    Low-rank generalization of the dense ChannelMixing block used in
    CycleNetMix.

    WHY THIS EXISTS: on Traffic (862 channels), CycleNetMix's dense
    channel -> hidden -> channel MLP has to compress 862 values down to
    mix_hidden (e.g. 128) before doing anything with them -- discarding
    roughly 85% of the per-timestep information before "mixing" even
    happens. At the same time, the dense weight matrix is free to fit
    ANY of the ~371,000 possible channel-pair interactions, when the real
    structure (per the dependency analysis and the traffic-forecasting
    literature) is that only a small, LOCAL subset of those pairs carry
    genuine signal -- most sensors are only meaningfully related to a
    handful of physically nearby ones. A dense matrix with ~11k training
    rows (Traffic's train set size) has both too little capacity to keep
    the full channel count AND too much freedom to fit spurious pairwise
    noise, which matches what was observed: CycleNetMix's Traffic gate
    opened to a small negative value and test error got WORSE, consistent
    with the branch fitting noise rather than real structure.

    HOW IT WORKS: instead of one dense channel -> hidden -> channel
    transform, the mixing weight is factorized into TWO small matrices:
        A: channel_size -> rank   (encode each channel into a small
                                    number of shared latent factors)
        B: rank -> channel_size   (decode those factors back out)
    with `rank` set small (e.g. 8-16) relative to channel_size. This
    forces the model to explain cross-channel dependency through a small
    number of shared underlying factors (plausible for road traffic: a
    handful of real congestion patterns/corridors that many sensors
    partially load onto) rather than through an unconstrained dense
    interaction matrix. It also cuts the parameter count drastically
    versus a dense hidden layer, directly reducing the branch's capacity
    to overfit.

    Same gating discipline as CycleNetMix: the branch's output is scaled
    by a learnable gate initialized to zero, so the model starts out
    identical to base CycleNet and only opens the low-rank mixing pathway
    if doing so reduces the training loss.
    """
    def __init__(self, channel_size, rank, dropout=0.1):
        super(LowRankChannelMixing, self).__init__()
        self.norm = nn.LayerNorm(channel_size)
        self.encode = nn.Linear(channel_size, rank, bias=False)   # A: channel_size -> rank
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.decode = nn.Linear(rank, channel_size, bias=True)    # B: rank -> channel_size
        self.dropout2 = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, seq_len, channels)
        residual = x
        h = self.norm(x)
        h = self.encode(h)          # (batch, seq_len, rank) -- compress to shared latent factors
        h = self.activation(h)
        h = self.dropout1(h)
        h = self.decode(h)          # (batch, seq_len, channels) -- expand back out
        h = self.dropout2(h)
        return residual + self.gate * h


class Model(nn.Module):
    """
    CycleNet + gated, LOW-RANK channel-mixing branch.

    Identical pipeline to CycleNetMix except the mixing block (step 3) is
    factorized through a small rank instead of a dense hidden layer:
        1. optional instance norm (RevIN-style)          -- unchanged
        2. remove learned cycle component                -- unchanged
        3. [LOW-RANK MIX] exchange information across channels on the
           residual, via a rank-constrained bottleneck
        4. channel-independent temporal backbone         -- unchanged
        5. add back cycle at future positions              -- unchanged
        6. optional instance denorm                        -- unchanged

    Extra config attributes (all optional, with safe defaults):
        mix_rank    : the shared low-rank dimension (default 16). Should
                      be much smaller than enc_in for large-channel-count
                      datasets like Traffic (862 channels); can be closer
                      to enc_in for small datasets like Weather (21
                      channels) if a wider bottleneck is preferred there.
        mix_layers  : number of stacked low-rank mixing blocks (default 1)
        mix_dropout : dropout inside each mixing block (default 0.1)
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

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        self.channel_mixing = nn.ModuleList([
            LowRankChannelMixing(self.enc_in, mix_rank, mix_dropout)
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

        # [LOW-RANK MIX] cross-channel information exchange via a small shared factor space
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