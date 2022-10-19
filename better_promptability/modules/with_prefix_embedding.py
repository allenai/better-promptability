import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class WithPrefixEmbedding(nn.Module):
    """
    From
    https://github.com/shmsw25/Channel-LM-Prompting/blob/cbbb92cc97039c73475ddf0db46896e9efeff3c1/model_util.py#L113
    """

    def __init__(self, orig_embed, expected_vocab_size, n_prefix):
        super().__init__()

        self.expected_vocab_size = expected_vocab_size
        orig_embed_len = orig_embed.weight.shape[0]
        assert expected_vocab_size <= orig_embed_len
        if expected_vocab_size < orig_embed_len:
            logger.warning(
                f"Embedding matrix will be resized from {orig_embed_len} to {expected_vocab_size}. "
                "This is expected for at least T5, and maybe some other models too. "
                "See https://github.com/huggingface/transformers/issues/4875#issuecomment-997299787"
            )

        self.embed = orig_embed
        self.new_embed = nn.Embedding(n_prefix, self.embed.embedding_dim)

        # following Lester et al. 2021 in initializing using the top 5000 random vocabs
        indices = np.random.permutation(range(5000))[:n_prefix]
        init_weight = self.embed.state_dict()["weight"][indices]
        self.new_embed._load_from_state_dict({"weight": init_weight}, "", None, True, [], [], "")

    def forward(self, input):
        return F.embedding(
            input,
            torch.cat([self.embed.weight[: self.expected_vocab_size], self.new_embed.weight], 0),
            self.embed.padding_idx,
            self.embed.max_norm,
            self.embed.norm_type,
            self.embed.scale_grad_by_freq,
            self.embed.sparse,
        )
