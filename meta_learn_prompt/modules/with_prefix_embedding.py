import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class WithPrefixEmbedding(nn.Module):
    """
    From https://github.com/shmsw25/Channel-LM-Prompting/blob/cbbb92cc97039c73475ddf0db46896e9efeff3c1/model_util.py#L113
    """

    def __init__(self, orig_embed, n_prefix):
        super().__init__()
        self.embed = orig_embed
        self.new_embed = nn.Embedding(n_prefix, self.embed.embedding_dim)

        # following Lester et al. 2021 in initializing using the top 5000 random vocabs
        indices = np.random.permutation(range(5000))[:n_prefix]
        init_weight = self.embed.state_dict()["weight"][indices]
        self.new_embed._load_from_state_dict({"weight": init_weight}, "", None, True, [], [], "")

    def forward(self, input):
        return F.embedding(
            input,
            torch.cat([self.embed.weight, self.new_embed.weight], 0),
            self.embed.padding_idx,
            self.embed.max_norm,
            self.embed.norm_type,
            self.embed.scale_grad_by_freq,
            self.embed.sparse,
        )
