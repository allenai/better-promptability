from __future__ import annotations
import logging
from typing import Any, Callable, IO, Optional, Union, Dict

import torch
from tango.common.lazy import Lazy
from tango.integrations.torch.optim import Optimizer
from transformers import T5ForConditionalGeneration

from ..data.config import Config
from ..data.prompt_data_module import PromptDataModule
from ..data.t0_multitask_data_module import T0MultiTaskDataModule
from ..modules.transformer import Transformer
from ..modules.with_prefix_embedding import WithPrefixEmbedding
from .model import Model
from .t5_with_prefix import T5WithPrefixConfig, T5ForConditionalGenerationWithPrefix

logger = logging.getLogger(__name__)


@Model.register("prefix_transformer")
@Model.register("prefix_transformer_from_checkpoint", constructor="load_from_checkpoint")
class PrefixTransformer(Model):
    def __init__(
        self,
        config: Config,
        dataset: PromptDataModule,
        transformer_model: str,
        optimizer: Optional[Lazy[Optimizer]] = None,
        epochs: int = 3,
        weight_decay: float = 0.0,
        accumulate_grad_batches: int = 1,
        warmup_steps: int = 0,
        train_full_model: bool = False,
        **transformer_kwargs,
    ):
        self.transformer_name = transformer_model
        self.train_full_model = train_full_model
        self.deep = dataset.deep

        super().__init__(
            config,
            dataset,
            optimizer=optimizer,
            epochs=epochs,
            weight_decay=weight_decay,
            accumulate_grad_batches=accumulate_grad_batches,
            warmup_steps=warmup_steps,
        )

        if not self.deep:
            self.transformer = Transformer(transformer_model, "seq2seq-lm", **transformer_kwargs)
        else:
            self.transformer = Transformer(
                transformer_model,
                "seq2seq-lm",
                config_cls=T5WithPrefixConfig,
                model_cls=T5ForConditionalGenerationWithPrefix,
                num_prefix=dataset.num_prefix,
                **transformer_kwargs,
            )
        transformer_model: T5ForConditionalGeneration = self.transformer.model
        assert isinstance(transformer_model, T5ForConditionalGeneration)

        if not self.train_full_model:
            for n, param in self.transformer.named_parameters():
                if n.startswith("model.encoder.prefix_") or n.startswith("model.decoder.prefix_"):
                    assert self.deep
                else:
                    param.requires_grad = False

        if not self.deep:
            transformer_model.set_input_embeddings(
                WithPrefixEmbedding(
                    transformer_model.shared,
                    self.dataset.tokenizer.vocab_size,
                    self.dataset.num_prefix,
                )
            )

    def unfreeze(self) -> dict[torch.nn.Parameter, bool]:
        orig_requires_grad = {}
        for param in self.transformer.parameters():
            orig_requires_grad[param] = param.requires_grad
            param.requires_grad = True
        return orig_requires_grad

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        return_dict = {}

        assert input_ids.shape == input_mask.shape and input_ids.dim() in (2, 3)
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
        scores = -loss.sum(-1) / (mask.sum(-1) + 1e-6)  # already masked in compute_loss()
        return scores

    def eval_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx=0,
        compute_loss=True,
    ) -> dict[str, Any]:
        if isinstance(self.dataset, T0MultiTaskDataModule):
            preds = self(batch)["logits"]
            split = self.dataset.dev_splits[dataloader_idx]
            for metric in self.metrics[split].values():
                metric(*metric.detach_tensors(preds, batch["target_ids"], batch["target_mask"]))
            return {}
        else:
            return super().eval_step(
                batch, batch_idx, dataloader_idx=dataloader_idx, compute_loss=False
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        """
        PyTorch's native optimizer state checkpoint logic is very fragile, so we also do it on our
        own. See https://github.com/pytorch/pytorch/issues/1489
        Also, when prompt-tuning, only stores prompt embedding in the checkpoint.
        """
        optimizer_states = self.optimizers(use_pl_optimizer=False).state
        if not self.train_full_model:
            weight_keys = (
                ["transformer.model.shared.new_embed.weight"]
                if not self.deep
                else [
                    k
                    for k in checkpoint["state_dict"].keys()
                    if k.startswith("transformer.model.encoder.prefix_")
                    or k.startswith("transformer.model.decoder.prefix_")
                ]
            )
            checkpoint["state_dict"] = {k: checkpoint["state_dict"][k] for k in weight_keys}

            name_to_param = {n: p for n, p in self.named_parameters()}
            states = {k: optimizer_states[name_to_param[k]] for k in weight_keys}
        else:
            param_to_name = {p: n for n, p in self.named_parameters()}
            states = {param_to_name[p]: states for p, states in optimizer_states.items()}
        checkpoint["custom_optimizer_states"] = states

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if any(k.startswith("model.") for k in checkpoint["state_dict"].keys()):
            # Unwrap the meta-learning model
            new_state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                assert k.startswith("model.")
                new_state_dict[k[len("model.") :]] = v
            checkpoint["state_dict"] = new_state_dict
        # TODO: optimizer states
        return super().on_load_checkpoint(checkpoint)

    def meta_learning_copy(self):
        new = PrefixTransformer(
            self.config,
            self.dataset,
            self.transformer_name,
            optimizer=self._optimizer,
            epochs=self.epochs,
            weight_decay=self.optimizer_kwargs["weight_decay"],
            accumulate_grad_batches=self.optimizer_kwargs["accumulate_grad_batches"],
            warmup_steps=self.optimizer_kwargs["warmup_steps"],
            train_full_model=self.train_full_model,
            deep=self.deep,
        )
        new.to(self.device)
        new.load_state_dict(self.state_dict())
        return new

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        optimizer: Optional[Lazy[Optimizer]] = None,
        **kwargs,
    ):
        # We need to tell tango the type of optimizer, or otherwise it will only give us a Params
        # object
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            optimizer=optimizer,
            **kwargs,
        )
