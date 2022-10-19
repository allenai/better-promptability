"""
Sorry, running this script as-is won't work. You'll have to modify HF's T5Attention's forward by
adding:
```
if key_value_states is None:  # self attention only
    import transformers
    getattr(transformers, "__my_secret_kv_property")["encoder_self" if not self.is_decoder else "decoder_self"].append((key_states, value_states))  # noqa: E501
```
"""

import sys
from pathlib import Path

from pytorch_lightning import seed_everything
import torch
import transformers
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from better_promptability.modules.transformer import Transformer  # noqa: E402
from better_promptability.modules.with_prefix_embedding import WithPrefixEmbedding  # noqa: E402


seed_everything(100)
transformers.__my_secret_kv_property = {"encoder_self": [], "decoder_self": []}


def main(model_name, num_prefix):
    num_prefix = int(num_prefix)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = Transformer(model_name, "seq2seq-lm")
    transformer.model.set_input_embeddings(
        WithPrefixEmbedding(transformer.model.shared, tokenizer.vocab_size, num_prefix)
    )
    transformer.cuda().eval()

    with torch.no_grad():
        input = (
            torch.arange(tokenizer.vocab_size, tokenizer.vocab_size + num_prefix)
            .unsqueeze(0)
            .cuda()
        )
        transformer(input, decoder_input_ids=input)
    assert all(
        len(kvs) == transformer.config.num_layers
        for kvs in transformers.__my_secret_kv_property.values()
    )
    kvs = {
        type: [[t.squeeze(0).detach().cpu() for t in (k, v)] for k, v in kvs]
        for type, kvs in transformers.__my_secret_kv_property.items()
    }
    # k/v: (num_heads, num_prefix, kv_dim)
    torch.save(kvs, f"kv_{model_name.replace('/', '_')}_{num_prefix}.pt")


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
