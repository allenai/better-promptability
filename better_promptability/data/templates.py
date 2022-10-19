from __future__ import annotations
import string
from typing import Any


def templatize(
    task: str, idx: int, example: dict[str, Any], label: Any, soft_only: bool = False
) -> tuple[str, str]:
    raise NotImplementedError("stale")
    from .super_glue_data_module import SUPER_GLUE_DATASETS

    # label is any task-defined label, e.g. 1/0, True/False, "entailment"/"non-entailment"
    if task in SUPER_GLUE_DATASETS:
        y, x = templatize_superglue(task, idx, example, label, soft_only=soft_only)
    else:
        if soft_only:
            y = get_sentence_classsification_verbalizers(task)[label]
        else:
            y = get_sentence_classsification_templates(task, idx, label)
        x = example["text"]
    return (x, y.strip())


def get_possible_labels(task, example):
    from .super_glue_data_module import SUPER_GLUE_DATASETS

    if task in SUPER_GLUE_DATASETS:
        if task == "boolq":
            return (0, 1)
        elif task == "cb":
            return (0, 1, 2)
        elif task == "rte":
            return (0, 1)
        elif task == "copa":
            return (0, 1)
        elif task == "wic":
            return (0, 1)
        elif task == "multirc":
            return (0, 1)
        elif task == "record":
            return example["entities"]
        else:
            raise NotImplementedError
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


def templatize_superglue(task, idx, example, label, soft_only=False):
    """
    Heavily references the design in https://arxiv.org/pdf/2009.07118.pdf
    """
    # fmt: off
    if task == "boolq":
        assert label in {0, 1}
        n_templates = 3 if soft_only else 4
        if 0 <= idx < n_templates:
            verbalizer = "Yes" if label == 1 else "No"
        elif idx < 2 * n_templates:
            verbalizer = "True" if label == 1 else "False"
            idx -= n_templates
        else:
            assert False

        passage = example["passage"]
        question = example["question"]
        if soft_only:
            templates = [
                (f"{verbalizer}", f"{passage} || {question}"),
                (f"{question} || {verbalizer}", f"{passage}"),
                (f"{passage} || {verbalizer}", f"{question}"),
            ]
        else:
            templates = [
                (f"{verbalizer}.", f"{passage} {question}?"),
                (f"Question: {question}? Answer: {verbalizer}.", f"{passage}"),
                (f"Based on the previous passage, {question}? {verbalizer}.", f"{passage}"),
                (f"{passage} {verbalizer}.", f"Based on the following passage, {question}?"),
            ]
        assert len(templates) == n_templates
        return templates[idx]
    elif task in {"cb", "rte"}:
        assert label in {0, 1, 2}
        verbalizer_a = {
            0: "Yes",
            1: "No",
            2: "Maybe",
        }[label]
        verbalizer_b = {
            0: "true",
            1: "not true",
            2: "maybe true",
        }[label]
        verbalizer_c = {
            0: "True",
            1: "False",
            2: "Neither",
        }[label]
        premise = example["premise"]
        hypothesis = example["hypothesis"].rstrip(string.punctuation)
        if soft_only:
            templates = [
                (f"{verbalizer_a}", f"{premise} || {hypothesis}"),
                (f"{hypothesis} || {verbalizer_a}", f"{premise}"),
                (f"{premise} || {verbalizer_a}", f"{hypothesis}"),
            ]
        else:
            templates = [
                (f"{verbalizer_a}.", f"{premise} {hypothesis}?"),
                (f"{verbalizer_a}, {premise}", f"{hypothesis}?"),
                (f'{verbalizer_a}, "{premise}"', f'"{hypothesis}"?'),
                (f"{verbalizer_a}. {premise}", f"{hypothesis}?"),
                (f'{verbalizer_a}. "{premise}"', f'"{hypothesis}"?'),
                (f"It is {verbalizer_b} that {hypothesis}.", f"{premise}"),
            ]
            if task == "cb":
                templates.append((f"True, False, or Neither? Answer: {verbalizer_c}.", f"{premise} Question: {hypothesis}."))  # noqa: E501
                templates.append((f"Question: {hypothesis}. True, False, or Neither? Answer: {verbalizer_c}.", f"{premise}"))  # noqa: E501
            elif task == "rte":
                templates.append((f"True or False? Answer: {verbalizer_c}.", f"{premise} Question: {hypothesis}."))  # noqa: E501
                templates.append((f"Question: {hypothesis}. True or False? Answer: {verbalizer_c}.", f"{premise}"))  # noqa: E501
        return templates[idx]
    elif task == "copa":
        assert label in {0, 1}
        premise = example["premise"].rstrip(string.punctuation)
        choice1 = example["choice1"].rstrip(string.punctuation)
        choice2 = example["choice2"].rstrip(string.punctuation)
        verbalizer = [choice1, choice2][label]
        connective = "so" if example["question"] == "effect" else "because"
        if soft_only:
            templates = [
                (f"{verbalizer}", f'{premise} || {connective} || {choice1} || {choice2}'),
                (f"{choice1} || {choice2} || {verbalizer}", f'{premise} || {connective}'),
                (f"{connective} || {choice1} || {choice2} || {verbalizer}", f'{premise}'),
                (f"{premise} || {verbalizer}", f'{connective} || {choice1} || {choice2}'),
            ]
        else:
            templates = [
                (f"{verbalizer}.", f'{premise}, {connective} "{choice1}." or "{choice2}."?'),
                (f"{verbalizer}.", f'{premise}, {connective} {choice1} or {choice2}?'),
                (f"{premise}, {connective} {verbalizer}.", f'"{choice1}." or "{choice2}."?'),
                (f"{premise}, {connective} {verbalizer}.", f'{choice1} or {choice2}?'),
            ]
        return templates[idx]
    elif task == "wic":
        assert label in {0, 1}
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        word = example["word"]
        verbalizer = "Yes" if label == 1 else "No"

        if soft_only:
            templates = [
                (f"{verbalizer}", f'{sentence1} || {sentence2} || {word}'),
                (f"{word} || {verbalizer}", f'{sentence1} || {sentence2}'),
            ]
        else:
            templates = [
                (f"{verbalizer}", f'"{sentence1}"/"{sentence2}". Similar sense of "{word}"?'),
                (f"{verbalizer}", f"{sentence1} {sentence2} Does {word} have the same meaning in both sentences?"),  # noqa: E501
                (f'Similar sense of "{word}"? {verbalizer}', f'"{sentence1}"/"{sentence2}".'),
                (f"Does {word} have the same meaning in both sentences? {verbalizer}", f"{sentence1} {sentence2}"),  # noqa: E501
            ]
        return templates[idx]
    elif task == "wsc":
        # This is a free form generation task which the channel model can't handle
        raise NotImplementedError
    elif task == "multirc":
        passage = example["paragraph"]
        question = example["question"]
        answer = example["answer"]

        assert label in {0, 1}
        n_templates = 4 if soft_only else 9
        if 0 <= idx < n_templates:
            verbalizer = "Yes" if label == 1 else "No"
        elif idx < 2 * n_templates:
            verbalizer = "True" if label == 1 else "False"
            idx -= n_templates
        else:
            assert False

        if soft_only:
            templates = [
                (f"{verbalizer}", f"{passage} || {question} || {answer}"),
                (f"{answer} || {verbalizer}", f"{passage} || {question}"),
                (f"{question} || {answer} || {verbalizer}", f"{passage}"),
                (f"{passage} || {verbalizer}", f"{question} || {answer}"),
            ]
        else:
            templates = [
                (f"{verbalizer}.", f"{passage} Question: {question} Is it {answer}?"),
                (f"Is it {answer}? {verbalizer}.", f"{passage} Question: {question}"),
                (f"Question: {question} Is it {answer}? {verbalizer}.", f"{passage}"),
                (f"{verbalizer}.", f'{passage} Question: {question} Is the correct answer "{answer}"?'),  # noqa: E501
                (f'Is the correct answer "{answer}"? {verbalizer}.', f"{passage} Question: {question}"),  # noqa: E501
                (f'Question: {question} Is the correct answer "{answer}"? {verbalizer}.', f"{passage}"),  # noqa: E501
                (f"{verbalizer}.", f'{passage} Based on the previous passage, {question} Is "{answer}" a correct answer?'),  # noqa: E501
                (f'Is "{answer}" a correct answer? {verbalizer}.', f"{passage} Based on the previous passage, {question}"),  # noqa: E501
                (f'Based on the previous passage, {question} Is "{answer}" a correct answer? {verbalizer}.', f"{passage}"),  # noqa: E501
            ]
        assert len(templates) == n_templates
        return templates[idx]
    elif task == "record":
        passage = example["passage"]
        query = example["query"]
        answer = label
        assert "@placeholder" in query
        if soft_only:
            return [
                (f"{answer}", f"{passage} || {query}"),
                (f"{query} || {answer}", f"{passage}"),
                (f"{passage} || {answer}", f"{query}"),
            ][idx]
        else:
            assert idx == 0
            query.replace("@placeholder", answer)
            return (query, passage)
    else:
        assert False
    # fmt: on
