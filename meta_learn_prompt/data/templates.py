import string
from typing import Any


SUPERGLUE_TASKS = {"boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"}


def templatize(task: str, idx: int, example: dict[str, Any], label: Any) -> tuple[str, str]:
    # label is any task-defined label, e.g. 1/0, True/False, "entailment"/"non-entailment"
    # Note that due the noisy channel model, "prefix" contains the label
    if task in SUPERGLUE_TASKS:
        prefix, input = templatize_superglue(task, idx, example, label)
    else:
        prefix = get_sentence_classsification_templates(task, idx, label)
        input = example["text"]
    return (prefix.strip(), " " + input)


def get_possible_labels(task):
    if task in SUPERGLUE_TASKS:
        raise NotImplementedError  # TODO: get this from datasets
    else:
        return list(range(len(get_sentence_classsification_verbalizers(task))))


def get_sentence_classsification_templates(task, idx, label):
    """
    From https://github.com/shmsw25/Channel-LM-Prompting/blob/cbbb92cc97039c73475ddf0db46896e9efeff3c1/util.py#L73
    """
    if task in ["sst-2", "sst-5", "mr", "cr"]:
        templates = ["A %s one . ", "It was %s . ", "All in all %s . ", "A %s piece . "]
    elif task in ["yelp_full", "yelp_binary", "amazon"]:
        templates = ["A %s one. ", "It was %s. ", "All in all %s. ", "A %s piece. "]
    elif task == "trec":
        templates = ["%s : ", "Q: %s : ", "Why %s ? ", "Answer: %s . "]
    elif task in ["agnews", "sogou", "dbpedia", "yahoo"]:
        templates = [
            "Topic: %s. ",
            "Subject: %s. ",
            "This is about %s. ",
            "It is about %s. ",
        ]
    elif task == "subj":
        templates = ["This is %s . ", "It's all %s . ", "It's %s . ", "Is it %s ? "]
    elif task == "cola":
        templates = ["This is %s .", "It is %s .", "You are %s .", "I am %s ."]
    else:
        raise NotImplementedError(task)

    verbalizers = get_sentence_classsification_verbalizers(task)
    return templates[idx] % verbalizers[label]


def get_sentence_classsification_verbalizers(task):
    if task in ["sst-2", "mr", "cr", "yelp_binary"]:
        verbalizers = ["terrible", "great"]
    elif task in ["sst-5", "yelp_full", "amazon"]:
        verbalizers = ["terrible", "bad", "okay", "good", "great"]
    elif task in ["agnews"]:
        verbalizers = ["World", "Sports", "Business", "Technology"]
    elif task in ["trec"]:
        verbalizers = [
            "Description",
            "Entity",
            "Expression",
            "Human",
            "Location",
            "Number",
        ]
    elif task in ["sogou"]:
        verbalizers = ["Sports", "Finance", "Entertainment", "Automobile", "Technology"]
    elif task in ["subj"]:
        verbalizers = ["subjective", "objective"]
    elif task in ["cola"]:
        verbalizers = ["not grammatical", "grammatical"]
    elif task in ["dbpedia"]:
        verbalizers = [
            "Company",
            "Educational Institution",
            "Artist",
            "Athlete",
            "Office Holder",
            "Mean of Transportation",
            "Building",
            "Natural Place",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "Written Work",
        ]
    elif task in ["yahoo"]:
        verbalizers = [
            "Society & Culture",
            "Science & Mathematics",
            "Health",
            "Education & Reference",
            "Computers & Internet",
            "Sports",
            "Business & Finance",
            "Entertainment & Music",
            "Family & Relationships",
            "Politics & Government",
        ]
    else:
        raise NotImplementedError(task)

    return verbalizers


def templatize_superglue(task, idx, example, label):
    """
    Heavily references the design in https://arxiv.org/pdf/2009.07118.pdf
    Returns (channel input, channel output)
    """
    # fmt: off
    if task == "boolq":
        n_templates = 4
        if 0 <= idx < n_templates:
            verbalizer = "Yes" if label else "No"
        elif idx < 2 * n_templates:
            verbalizer = "True" if label else "False"
            idx -= n_templates
        else:
            assert False

        passage = example["passage"]
        question = example["question"]
        templates = [
            (f"{verbalizer}.", f"{passage} {question}?"),
            (f"Question: {question}? Answer: {verbalizer}.", f"{passage}"),
            (f"Based on the previous passage, {question}? {verbalizer}.", f"{passage}"),
            (f"{passage} {verbalizer}.", f"Based on the following passage, {question}?"),
        ]
        assert len(templates) == n_templates
        return templates[idx]
    elif task in {"cb", "rte"}:
        verbalizer_a = {
            "entailment": "Yes",
            "not_entailment": "No",
            "contradiction": "No",
            "neutral": "Maybe",
        }[label]
        verbalizer_b = {
            "entailment": "true",
            "not_entailment": "not true",
            "contradiction": "not true",
            "neutral": "maybe true",
        }[label]
        verbalizer_c = {
            "entailment": "True",
            "not_entailment": "False",
            "contradiction": "False",
            "neutral": "Neither",
        }[label]
        premise = example["premise"]
        hypothesis = example["hypothesis"].rstrip(string.punctuation)
        templates = [
            (f"{verbalizer_a}.", f"{premise} {hypothesis}?"),
            (f"{verbalizer_a}, {premise}", f"{hypothesis}?"),
            (f'{verbalizer_a}, "{premise}"', f'"{hypothesis}"?'),
            (f"{verbalizer_a}. {premise}", f"{hypothesis}?"),
            (f'{verbalizer_a}. "{premise}"', f'"{hypothesis}"?'),
            (f"It is {verbalizer_b} that {hypothesis}.", f"{premise}"),
        ]
        if task == "cb":
            templates.append((f"True, False, or Neither? Answer: {verbalizer_c}.", f"{premise} Question: {hypothesis}."))
            templates.append((f"Question: {hypothesis}. True, False, or Neither? Answer: {verbalizer_c}.", f"{premise}"))
        elif task == "rte":
            templates.append((f"True or False? Answer: {verbalizer_c}.", f"{premise} Question: {hypothesis}."))
            templates.append((f"Question: {hypothesis}. True or False? Answer: {verbalizer_c}.", f"{premise}"))
        return templates[idx]
    elif task == "copa":
        premise = example["premise"].rstrip(string.punctuation)
        choice1 = example["choice1"].rstrip(string.punctuation)
        choice2 = example["choice2"].rstrip(string.punctuation)
        # TODO: random swap
        verbalizer = [choice1, choice2][label]
        connective = "so" if example["question"] == "effect" else "because"
        templates = [
            (f"{verbalizer}.", f'{premise}, {connective} "{choice1}." or "{choice2}."?'),
            (f"{verbalizer}.", f'{premise}, {connective} {choice1} or {choice2}?'),
            (f"{premise}, {connective} {verbalizer}.", f'"{choice1}." or "{choice2}."?'),
            (f"{premise}, {connective} {verbalizer}.", f'{choice1} or {choice2}?'),
        ]
        return templates[idx]
    elif task == "wic":
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        # TODO: random swap
        word = example["word"]
        verbalizer = "Yes" if label else "No"

        templates = [
            (f"{verbalizer}", f'"{sentence1}"/"{sentence2}". Similar sense of "{word}"?'),
            (f"{verbalizer}", f"{sentence1} {sentence2} Does {word} have the same meaning in both sentences?"),
            (f'Similar sense of "{word}"? {verbalizer}', f'"{sentence1}"/"{sentence2}".'),
            (f"Does {word} have the same meaning in both sentences? {verbalizer}", f"{sentence1} {sentence2}"),
        ]
        return templates[idx]
    elif task == "wsc":
        # This is a free form generation task which the channel model can't handle
        raise NotImplementedError
    elif task == "multirc":
        passage = example["passage"]
        question = example["question"]
        answer = example["answer"]

        assert label in {0, 1}
        n_templates = 9
        if 0 <= idx < n_templates:
            verbalizer = "Yes" if label == 1 else "No"
        elif idx < 2 * n_templates:
            verbalizer = "True" if label == 1 else "False"
            idx -= n_templates
        else:
            assert False

        templates = [
            (f"{verbalizer}.", f"{passage} Question: {question} Is it {answer}?"),
            (f"Is it {answer}? {verbalizer}.", f"{passage} Question: {question}"),
            (f"Question: {question} Is it {answer}? {verbalizer}.", f"{passage}"),
            (f"{verbalizer}.", f'{passage} Question: {question} Is the correct answer "{answer}"?'),
            (f'Is the correct answer "{answer}"? {verbalizer}.', f"{passage} Question: {question}"),
            (f'Question: {question} Is the correct answer "{answer}"? {verbalizer}.', f"{passage}"),
            (f"{verbalizer}.", f'{passage} Based on the previous passage, {question} Is "{answer}" a correct answer?'),
            (f'Is "{answer}" a correct answer? {verbalizer}.', f"{passage} Based on the previous passage, {question}"),
            (f'Based on the previous passage, {question} Is "{answer}" a correct answer? {verbalizer}.', f"{passage}"),
        ]
        assert len(templates) == n_templates
        return templates[idx]
    elif task == "record":
        passage = example["passage"]
        query = example["query"]
        answer = example["answer"]
        assert "@placeholder" in query
        query.replace("@placeholder", answer)
        return (query, passage)
    else:
        assert False
    # fmt: on
