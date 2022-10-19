import pytest
from transformers.models import t5 as hf_t5
from better_promptability.modules.transformer import Transformer


@pytest.fixture(scope="module")
def model_name():
    return "google/t5-small-lm-adapt"


@pytest.fixture(scope="module")
def tokenizer(model_name):
    return hf_t5.T5Tokenizer.from_pretrained(model_name)


@pytest.mark.parametrize(
    "task",
    [
        "seq2seq-lm",
    ],
)
def test_transformer(task: str, model_name: str, tokenizer: hf_t5.T5Tokenizer):

    model = Transformer(model_name, task=task)

    input_ids = tokenizer(
        ["The <extra_id_0> walks in <extra_id_1> park", "The <extra_id_0> barked"],
        return_tensors="pt",
        padding=True,
    ).input_ids
    assert input_ids.tolist() == [
        [37, 32099, 10681, 16, 32098, 2447, 1],
        [37, 32099, 1207, 5100, 1, 0, 0],
    ]

    attention_mask = ~(input_ids == 0)

    labels = tokenizer(
        ["<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", "<extra_id_0> dog"],
        return_tensors="pt",
        padding=True,
    ).input_ids
    assert labels.tolist() == [
        [32099, 5295, 1782, 32098, 8, 32097, 1],
        [32099, 1782, 1, 0, 0, 0, 0],
    ]

    decoder_attention_mask = ~(labels == 0)

    output = model.forward(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask,
    )

    assert output.logits is not None
