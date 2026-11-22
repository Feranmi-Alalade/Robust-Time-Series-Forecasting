# import math
# import numpy as np
# import torch
# import torch.nn as nn


# # ============================================================
# # RECURRENT CYCLE
# # ============================================================

# class RecurrentCycle(nn.Module):
#     def __init__(self, cycle_len, channel_size):
#         super().__init__()

#         self.cycle_len = cycle_len

#         self.data = nn.Parameter(
#             torch.zeros(cycle_len, channel_size)
#         )

#     def forward(self, index, length):

#         gather_index = (
#             index.view(-1, 1)
#             + torch.arange(
#                 length,
#                 device=index.device
#             ).view(1, -1)
#         ) % self.cycle_len

#         return self.data[gather_index]


# # ============================================================
# # SPARSE PAIRWISE CHANNEL MIXING
# # ============================================================

# class PairwiseMixing(nn.Module):
#     """
#     mask[i, j] = 1 means:

#         source channel j -> target channel i

#     Only selected channel pairs are allowed to interact.
#     """

#     def __init__(self, channels, mask):
#         super().__init__()

#         # Fixed D x D dependency mask
#         self.register_buffer(
#             "mask",
#             mask
#         )

#         # Learnable strength of each source-target relationship
#         self.weight = nn.Parameter(
#             torch.empty(channels, channels)
#         )

#         nn.init.xavier_uniform_(self.weight)

#         # Mixing branch initially contributes nothing
#         self.gate = nn.Parameter(
#             torch.zeros(1)
#         )

#     def forward(self, x):

#         # x: B x L x D

#         # Keep only selected source-target relationships
#         weight = self.weight * self.mask

#         # Apply the same D x D pairwise mixing at every timestep
#         mixed = torch.matmul(
#             x,
#             weight.T
#         )

#         # Gated residual connection
#         return x + self.gate * mixed


# # ============================================================
# # CYCLENET PAIR MIX
# # ============================================================

# class Model(nn.Module):

#     def __init__(self, configs):
#         super().__init__()

#         # Original CycleNet settings
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.cycle_len = configs.cycle
#         self.model_type = configs.model_type
#         self.d_model = configs.d_model
#         self.use_revin = configs.use_revin

#         # ====================================================
#         # PAIRWISE MIXING SETTINGS
#         # ====================================================

#         score_path = configs.dependency_score_path

#         self.mix_ratio = getattr(
#             configs,
#             "mix_ratio",
#             0.4
#         )

#         # Number of OTHER channels selected for each target
#         self.k = math.ceil(
#             self.mix_ratio * (self.enc_in - 1)
#         )


#         # ====================================================
#         # LOAD D x D DEPENDENCY SCORE MATRIX
#         # ====================================================

#         scores = np.load(
#             score_path
#         )

#         scores = torch.tensor(
#             scores,
#             dtype=torch.float32
#         )


#         # ====================================================
#         # BUILD BINARY DEPENDENCY MASK
#         # ====================================================

#         # Do not allow a channel to select itself
#         scores.fill_diagonal_(
#             float("-inf")
#         )

#         # For every target channel, select the strongest
#         # source channels according to the dependency scores
#         top_sources = torch.topk(
#             scores,
#             k=self.k,
#             dim=1
#         ).indices

#         # D x D binary mask
#         mask = torch.zeros(
#             self.enc_in,
#             self.enc_in
#         )

#         mask.scatter_(
#             1,
#             top_sources,
#             1.0
#         )


#         # ====================================================
#         # LEARNABLE RECURRENT CYCLE
#         # ====================================================

#         self.cycleQueue = RecurrentCycle(
#             self.cycle_len,
#             self.enc_in
#         )


#         # ====================================================
#         # SPARSE PAIRWISE MIXER
#         # ====================================================

#         self.channel_mixing = PairwiseMixing(
#             self.enc_in,
#             mask
#         )


#         # ====================================================
#         # ORIGINAL CYCLENET TEMPORAL BACKBONE
#         # ====================================================

#         if self.model_type == "linear":

#             self.model = nn.Linear(
#                 self.seq_len,
#                 self.pred_len
#             )

#         elif self.model_type == "mlp":

#             self.model = nn.Sequential(

#                 nn.Linear(
#                     self.seq_len,
#                     self.d_model
#                 ),

#                 nn.ReLU(),

#                 nn.Linear(
#                     self.d_model,
#                     self.pred_len
#                 )
#             )

#         else:

#             raise ValueError(
#                 "model_type must be 'linear' or 'mlp'"
#             )


#     def forward(self, x, cycle_index):

#         # ====================================================
#         # 1. INSTANCE NORMALIZATION
#         # ====================================================

#         if self.use_revin:

#             seq_mean = torch.mean(
#                 x,
#                 dim=1,
#                 keepdim=True
#             )

#             seq_var = (
#                 torch.var(
#                     x,
#                     dim=1,
#                     keepdim=True
#                 )
#                 + 1e-5
#             )

#             x = (
#                 x - seq_mean
#             ) / torch.sqrt(seq_var)


#         # ====================================================
#         # 2. REMOVE LEARNED CYCLE
#         # ====================================================

#         x = x - self.cycleQueue(
#             cycle_index,
#             self.seq_len
#         )


#         # ====================================================
#         # 3. SELECTIVE PAIRWISE CHANNEL MIXING
#         # ====================================================

#         x = self.channel_mixing(
#             x
#         )


#         # ====================================================
#         # 4. ORIGINAL CYCLENET TEMPORAL FORECASTING
#         # ====================================================

#         y = self.model(
#             x.permute(0, 2, 1)
#         ).permute(0, 2, 1)


#         # ====================================================
#         # 5. ADD FUTURE CYCLE
#         # ====================================================

#         future_cycle_index = (
#             cycle_index
#             + self.seq_len
#         ) % self.cycle_len

#         y = y + self.cycleQueue(
#             future_cycle_index,
#             self.pred_len
#         )


#         # ====================================================
#         # 6. INSTANCE DENORMALIZATION
#         # ====================================================

#         if self.use_revin:

#             y = (
#                 y
#                 * torch.sqrt(seq_var)
#                 + seq_mean
#             )


#         return y

import math
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# RECURRENT CYCLE
# ============================================================

class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super().__init__()

        self.cycle_len = cycle_len

        self.data = nn.Parameter(
            torch.zeros(cycle_len, channel_size)
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


# ============================================================
# SPARSE PAIRWISE CHANNEL MIXING
# ============================================================

class PairwiseMixing(nn.Module):
    """
    mask[i, j] = 1 means:

        source channel j -> target channel i

    Only selected channel pairs are allowed to interact.
    """

    def __init__(self, channels, mask):
        super().__init__()

        self.register_buffer(
            "mask",
            mask
        )

        self.weight = nn.Parameter(
            torch.empty(channels, channels)
        )

        nn.init.xavier_uniform_(
            self.weight
        )

        self.gate = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, x):

        # x: B x L x D

        weight = (
            self.weight
            * self.mask
        )

        mixed = torch.matmul(
            x,
            weight.T
        )

        return (
            x
            + self.gate * mixed
        )


# ============================================================
# CYCLENET PAIR MIX
# ============================================================

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        # ====================================================
        # ORIGINAL CYCLENET SETTINGS
        # ====================================================

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin


        # ====================================================
        # PAIRWISE MIXING SETTINGS
        # ====================================================

        score_path = configs.dependency_score_path

        if score_path is None:
            raise ValueError(
                "dependency_score_path must be provided "
                "for CycleNetPairMix"
            )

        self.mix_ratio = getattr(
            configs,
            "mix_ratio",
            0.4
        )

        # ----------------------------------------------------
        # Optuna can tune mix_k directly.
        #
        # If mix_k is not provided, fall back to the original
        # ratio-based calculation so old experiments continue
        # to work unchanged.
        # ----------------------------------------------------

        mix_k = getattr(
            configs,
            "mix_k",
            None
        )

        if mix_k is not None:

            self.k = int(
                mix_k
            )

        else:

            self.k = math.ceil(
                self.mix_ratio
                * (self.enc_in - 1)
            )

        # Safety:
        # at least one other channel
        # at most D - 1 other channels

        self.k = max(
            1,
            min(
                self.k,
                self.enc_in - 1
            )
        )


        # ====================================================
        # LOAD D x D DEPENDENCY SCORE MATRIX
        # ====================================================

        scores = np.load(
            score_path
        )

        if scores.shape != (
            self.enc_in,
            self.enc_in
        ):

            raise ValueError(
                f"Dependency matrix shape "
                f"{scores.shape} does not match "
                f"enc_in={self.enc_in}"
            )

        scores = torch.tensor(
            scores,
            dtype=torch.float32
        )


        # ====================================================
        # BUILD BINARY DEPENDENCY MASK
        # ====================================================

        # A channel cannot select itself.
        scores.fill_diagonal_(
            float("-inf")
        )

        # For each target channel, keep the k strongest
        # source channels according to dependency score.
        top_sources = torch.topk(
            scores,
            k=self.k,
            dim=1
        ).indices

        mask = torch.zeros(
            self.enc_in,
            self.enc_in,
            dtype=torch.float32
        )

        mask.scatter_(
            1,
            top_sources,
            1.0
        )


        # ====================================================
        # LEARNABLE RECURRENT CYCLE
        # ====================================================

        self.cycleQueue = RecurrentCycle(
            self.cycle_len,
            self.enc_in
        )


        # ====================================================
        # SPARSE PAIRWISE MIXER
        # ====================================================

        self.channel_mixing = PairwiseMixing(
            self.enc_in,
            mask
        )


        # ====================================================
        # ORIGINAL CYCLENET TEMPORAL BACKBONE
        # ====================================================

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

        else:

            raise ValueError(
                "model_type must be 'linear' or 'mlp'"
            )


    def forward(self, x, cycle_index):

        # ====================================================
        # 1. INSTANCE NORMALIZATION
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
            ) / torch.sqrt(
                seq_var
            )


        # ====================================================
        # 2. REMOVE LEARNED CYCLE
        # ====================================================

        x = x - self.cycleQueue(
            cycle_index,
            self.seq_len
        )


        # ====================================================
        # 3. SELECTIVE PAIRWISE CHANNEL MIXING
        # ====================================================

        x = self.channel_mixing(
            x
        )


        # ====================================================
        # 4. ORIGINAL CYCLENET TEMPORAL FORECASTING
        # ====================================================

        y = self.model(
            x.permute(
                0,
                2,
                1
            )
        ).permute(
            0,
            2,
            1
        )


        # ====================================================
        # 5. ADD FUTURE CYCLE
        # ====================================================

        future_cycle_index = (
            cycle_index
            + self.seq_len
        ) % self.cycle_len

        y = y + self.cycleQueue(
            future_cycle_index,
            self.pred_len
        )


        # ====================================================
        # 6. INSTANCE DENORMALIZATION
        # ====================================================

        if self.use_revin:

            y = (
                y
                * torch.sqrt(
                    seq_var
                )
                + seq_mean
            )

        return y