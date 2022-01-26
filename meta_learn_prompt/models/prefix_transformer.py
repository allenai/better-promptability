from __future__ import annotations
import logging
import os
import pickle
from typing import Any, Callable, IO, Optional, Union

import torch
from tango.common.lazy import Lazy
from tango.integrations.torch.optim import Optimizer
from transformers import T5ForConditionalGeneration

from ..data.config import Config
from ..data.prompt_data_module import PromptDataModule
from ..modules.transformer import Transformer
from ..modules.with_prefix_embedding import WithPrefixEmbedding
from ..train.optim import load_adafactor_state, resolve_optimizer_conf
from .model import Model

logger = logging.getLogger(__name__)


@Model.register("prefix_transformer")
class PrefixTransformer(Model):
    def __init__(
        self,
        config: Config,
        dataset: PromptDataModule,
        transformer_model: str,
        optimizer: Optional[Lazy[Optimizer]] = None,
        scheduler: Optional[str] = None,
        epochs: int = 3,
        weight_decay: float = 0.0,
        accumulate_grad_batches: int = 1,
        warmup_steps: int = 0,
        lr_scheduler_total_steps: Optional[int] = None,
        optstates_dir: Optional[str] = "/net/nfs2.allennlp/zhaofengw/optstates",
        load_opt_states: bool = True,
        train_full_model: bool = False,
        **transformer_kwargs,
    ):
        self.transformer_name = transformer_model
        self.optstates_dir = optstates_dir
        self.load_opt_states = load_opt_states
        self.train_full_model = train_full_model

        super().__init__(
            config,
            dataset,
            optimizer=optimizer,
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

        if not self.train_full_model:
            for param in self.transformer.parameters():
                param.requires_grad = False

        transformer_model.set_input_embeddings(
            WithPrefixEmbedding(
                transformer_model.shared, self.dataset.tokenizer.vocab_size, self.dataset.num_prefix
            )
        )

    def unfreeze(self):
        for param in self.transformer.parameters():
            param.requires_grad = True

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
        scores = -loss.sum(-1) / (mask.sum(-1) + 1e-6)  # already masked in compute_loss()
        return scores

    def eval_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx=0,
        compute_loss=True,
    ) -> dict[str, Any]:
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
            weight_key = "transformer.model.shared.new_embed.weight"
            checkpoint["state_dict"] = {weight_key: checkpoint["state_dict"][weight_key]}

            name_to_param = {n: p for n, p in self.named_parameters()}
            states = {weight_key: optimizer_states[name_to_param[weight_key]]}
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

    def configure_optimizers(
        self, load_opt_states: Optional[bool] = None
    ) -> Union[list[Optimizer], tuple[list[Optimizer], list[dict]]]:
        opt_conf = super().configure_optimizers()

        if load_opt_states is None:
            load_opt_states = self.load_opt_states
        if self._optimizer._params["type"] == "adafactor" and load_opt_states:  # type: ignore
            assert self.optstates_dir is not None
            optstates_path = os.path.join(self.optstates_dir, self.transformer_name.split("/")[-1])
            optstates = pickle.load(open(optstates_path, "rb"))
            optimizer = resolve_optimizer_conf(opt_conf)
            load_adafactor_state(self.transformer.model, optimizer, optstates)

        return opt_conf

    def meta_learning_copy(self):
        new = PrefixTransformer(
            self.config,
            self.dataset,
            self.transformer_name,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            epochs=self.epochs,
            weight_decay=self.optimizer_kwargs["weight_decay"],
            accumulate_grad_batches=self.optimizer_kwargs["accumulate_grad_batches"],
            warmup_steps=self.optimizer_kwargs["warmup_steps"],
            lr_scheduler_total_steps=self.optimizer_kwargs["lr_scheduler_total_steps"],
            optstates_dir=self.optstates_dir,
            load_opt_states=self.load_opt_states,
            train_full_model=self.train_full_model,
        )
        new.to(self.device)
        new.load_state_dict(self.state_dict())
        return new

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[dict[str, str], str, torch.device, int, Callable]] = None,
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


Model.register("prefix_transformer_from_checkpoint")(PrefixTransformer.load_from_checkpoint)
PrefixTransformer.register("from_checkpoint")(PrefixTransformer.load_from_checkpoint)
