from __future__ import annotations
import logging
from typing import Any, Optional

import torch
from tango.common.lazy import Lazy
from tango.integrations.pytorch_lightning.model import LightningModule
from tango.integrations.torch.optim import Optimizer
from transformers import T5ForConditionalGeneration

from ..data.config import Config
from ..data.prompt_data_module import PromptDataModule
from ..modules.transformer import Transformer
from ..modules.with_prefix_embedding import WithPrefixEmbedding
from .model import Model

logger = logging.getLogger(__name__)


@LightningModule.register("prefix_transformer")
class PrefixTransformer(Model):
    def __init__(
        self,
        config: Config,
        dataset: PromptDataModule,
        transformer_model: str,
        optimizer: Lazy[Optimizer],
        epochs: int = 3,
        weight_decay: float = 0.0,
        accumulate_grad_batches: int = 1,
        warmup_steps: int = 0,
        lr_scheduler_total_steps: Optional[int] = None,
        **transformer_kwargs,
    ):
        super().__init__(
            config,
            dataset,
            optimizer,
            epochs,
            # lr,
            weight_decay,
            accumulate_grad_batches,
            # adam_epsilon,
            warmup_steps,
            lr_scheduler_total_steps,
        )

        self.transformer = Transformer(transformer_model, "seq2seq-lm", **transformer_kwargs)
        transformer_model: T5ForConditionalGeneration = self.transformer.model
        assert isinstance(transformer_model, T5ForConditionalGeneration)

        for param in self.transformer.parameters():
            param.requires_grad = False

        transformer_model.set_input_embeddings(
            WithPrefixEmbedding(transformer_model.shared, self.dataset.num_prefix)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        assert input_ids.shape == attention_mask.shape and input_ids.dim() in (2, 3)
        return_dict = {}

        # if not self.training:  # for inference we have an additional dimension for classes
        # TODO: confirm logic.
        orig_shape = input_ids.shape  # bs x num_classes x seq_len
        input_ids = input_ids.reshape(-1, orig_shape[-1])
        attention_mask = attention_mask.reshape(-1, orig_shape[-1])
        decoder_input_ids = batch["targets"]
        decoder_attention_mask = batch["targets_mask"]
        orig_decoder_shape = decoder_input_ids.shape
        decoder_input_ids = decoder_input_ids.reshape(-1, orig_decoder_shape[-1])
        decoder_attention_mask = decoder_attention_mask.reshape(-1, orig_decoder_shape[-1])

        logits = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        ).logits
        logits = logits.reshape(*(orig_shape + (-1,)))
        return_dict["logits"] = logits

        return return_dict

    def get_predictions(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Input:
            logits: (bsz, num_classes, seq_len, vocab_size)
        Output:
            scores: (bsz, num_classes)
        """
        mask = batch["targets_mask"]  # (bsz, num_classes, seq_len)
        loss = self.compute_loss(logits, batch["targets"], mask, reduce=False)
        scores = -loss.sum(-1) / mask.sum(-1)  # already masked in compute_loss()
        return scores

    def eval_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        mode: str,
        dataloader_idx=0,
        compute_loss=True,
    ) -> dict[str, Any]:
        return super().eval_step(
            batch, batch_idx, mode, dataloader_idx=dataloader_idx, compute_loss=False
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        weight_key = "transformer.model.shared.new_embed.weight"
        print(checkpoint["state_dict"].keys())
        checkpoint["state_dict"] = {weight_key: checkpoint["state_dict"][weight_key]}


# @Step.register("get_model")
# class GetModel(Step):
#     DETERMINISTIC = True
#     CACHEABLE = False

#     def run(
#         self,
#         model: Lazy[PrefixTransformer],
#         config: Config,
#         dataset: FewShotDataModule,
#     ) -> PrefixTransformer:

#         return model.construct(config=config, dataset=dataset)
