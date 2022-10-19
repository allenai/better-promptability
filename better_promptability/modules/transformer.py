import logging

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
)


logger = logging.getLogger(__name__)


TASKS = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "causal-lm": AutoModelForCausalLM,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
    "seq2seq-lm": AutoModelForSeq2SeqLM,
}


class Transformer(torch.nn.Module):
    def __init__(
        self,
        transformer_model: str,
        task: str,
        trainable=True,
        config_cls=AutoConfig,
        model_cls=None,
        **config_kwargs,
    ):
        super().__init__()

        config_args = dict(config_kwargs)
        if task == "base":  # TODO: this might break models that don't support this flag
            config_args["add_pooling_layer"] = False
        self.config = config_cls.from_pretrained(transformer_model, **config_args)
        model_cls = model_cls if model_cls is not None else TASKS[task]
        self.model = model_cls.from_pretrained(transformer_model, config=self.config)

        if not trainable:  # TODO: support this
            assert task == "base", "No support for freezing the backbone for headed tasks yet"
        self.trainable = trainable

    def forward(self, *args, **kwargs):
        if "attention_mask" in kwargs:  # `transformers` doesn't take bool masks which is crazy
            kwargs["attention_mask"] = kwargs["attention_mask"].float()
        if "decoder_attention_mask" in kwargs:
            kwargs["decoder_attention_mask"] = kwargs["decoder_attention_mask"].float()
        # If grad was previous disabled (e.g., in eval), don't change it
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            return self.model(*args, **kwargs)
