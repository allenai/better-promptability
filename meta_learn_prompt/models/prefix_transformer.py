from __future__ import annotations
import logging
import os
import pickle
import re
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
        scheduler: Optional[str] = None,
        epochs: int = 3,
        weight_decay: float = 0.0,
        accumulate_grad_batches: int = 1,
        warmup_steps: int = 0,
        lr_scheduler_total_steps: Optional[int] = None,
        optstates_dir: Optional[str] = "/net/nfs2.allennlp/zhaofengw/optstates",
        **transformer_kwargs,
    ):
        self.transformer_name = transformer_model
        self.optstates_dir = optstates_dir

        super().__init__(
            config,
            dataset,
            optimizer,
            scheduler=scheduler,
            epochs=epochs,
            # lr,
            weight_decay=weight_decay,
            accumulate_grad_batches=accumulate_grad_batches,
            # adam_epsilon,
            warmup_steps=warmup_steps,
            lr_scheduler_total_steps=lr_scheduler_total_steps,
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
        input_mask = batch["input_mask"]
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        return_dict = {}

        assert input_ids.shape == input_mask.shape and input_ids.dim() in (2, 3)
        # assert self.training == (input_ids.dim() == 2)
        if not self.training:  # for inference we have an additional dimension for classes
            orig_shape = input_ids.shape  # bs x num_classes x seq_len
            input_ids = input_ids.reshape(-1, orig_shape[-1])
            input_mask = input_mask.reshape(-1, orig_shape[-1])

            orig_decoder_shape = target_ids.shape
            target_ids = target_ids.reshape(-1, orig_decoder_shape[-1])
            target_mask = target_mask.reshape(-1, orig_decoder_shape[-1])

        logits = self.transformer(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=target_ids,
            decoder_attention_mask=target_mask,
        ).logits

        if not self.training:
            logits = logits.reshape(*(orig_decoder_shape + (-1,)))
        return_dict["logits"] = logits

        return return_dict

    def get_predictions(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Input:
            logits: (bsz, num_classes, seq_len, vocab_size)
        Output:
            scores: (bsz, num_classes)
        """
        mask = batch["target_mask"]  # (bsz, num_classes, seq_len)
        loss = self.compute_loss(logits, batch["target_ids"], mask, reduce=False)
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

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        opt_conf = super().configure_optimizers()

        if self._optimizer._params["type"] == "adafactor":
            assert self.optstates_dir is not None
            optstates_path = os.path.join(self.optstates_dir, self.transformer_name.split("/")[-1])
            optstates = pickle.load(open(optstates_path, "rb"))

            if (
                isinstance(opt_conf, (list, tuple))
                and len(opt_conf) == 2
                and isinstance(opt_conf[0][0], Optimizer)
            ):
                # optimizers + schedulers
                optimizer = opt_conf[0][0]
            else:
                optimizer = opt_conf[0]

            for param_name, states in optstates.items():
                name = param_name.split("/")
                pointer = self.transformer.model
                # Following the logic at https://github.com/huggingface/transformers/blob/027074f4d0503e4fc077beb069e651435979b7b2/src/transformers/models/t5/modeling_t5.py#L116
                for m_name in name:
                    if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                        scope_names = re.split(r"_(\d+)", m_name)
                    else:
                        scope_names = [m_name]
                    if scope_names[0] in ["kernel", "scale", "embedding"]:
                        pointer = getattr(pointer, "weight")
                    elif scope_names[0] == "self_attention":
                        pointer = getattr(pointer, "layer")
                        pointer = pointer[0]
                    elif scope_names[0] == "enc_dec_attention":
                        pointer = getattr(pointer, "layer")
                        pointer = pointer[1]
                    elif scope_names[0] == "dense_relu_dense":
                        pointer = getattr(pointer, "layer")
                        pointer = pointer[2]
                    elif scope_names[0] == "rms_norm":
                        if hasattr(pointer, "layer_norm"):
                            pointer = getattr(pointer, "layer_norm")
                        elif hasattr(pointer, "final_layer_norm"):
                            pointer = getattr(pointer, "final_layer_norm")
                    elif scope_names[0] == "scale":
                        pointer = getattr(pointer, "weight")
                    elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                        pointer = getattr(pointer, "bias")
                    elif scope_names[0] == "squad":
                        pointer = getattr(pointer, "classifier")
                    elif scope_names[0] == "decoder" and name[1] == "logits":
                        continue
                    elif scope_names[0] == "logits":
                        pointer = getattr(pointer, "lm_head")
                    elif (
                        scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit()
                    ):
                        pointer = getattr(pointer, f"wi_{scope_names[1]}")
                        continue
                    else:
                        pointer = getattr(pointer, scope_names[0])
                        if isinstance(pointer, WithPrefixEmbedding):
                            pointer = pointer.embed
                    if len(scope_names) >= 2:
                        num = int(scope_names[1])
                        pointer = pointer[num]
                if scope_names[0] not in ["kernel", "scale", "embedding"]:
                    pointer = getattr(pointer, "weight")
                assert (("vr" in states) == ("vc" in states)) and (
                    ("vr" in states) != ("v" in states)
                )
                if "vr" in states:
                    optimizer.state[pointer]["exp_avg_sq_row"] = states["vr"]
                    optimizer.state[pointer]["exp_avg_sq_col"] = states["vc"]
                else:
                    optimizer.state[pointer]["exp_avg_sq"] = states["v"]
                optimizer.state[pointer]["step"] = 0

        return opt_conf


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
