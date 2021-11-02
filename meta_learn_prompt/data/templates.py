def get_templates(task, idx):
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

    if task in ["sst-2", "mr", "cr", "yelp_binary"]:
        label_words = ["terrible", "great"]
    elif task in ["sst-5", "yelp_full", "amazon"]:
        label_words = ["terrible", "bad", "okay", "good", "great"]
    elif task in ["agnews"]:
        label_words = ["World", "Sports", "Business", "Technology"]
    elif task in ["trec"]:
        label_words = [
            "Description",
            "Entity",
            "Expression",
            "Human",
            "Location",
            "Number",
        ]
    elif task in ["sogou"]:
        label_words = ["Sports", "Finance", "Entertainment", "Automobile", "Technology"]
    elif task in ["subj"]:
        label_words = ["subjective", "objective"]
    elif task in ["cola"]:
        label_words = ["not grammatical", "grammatical"]
    elif task in ["dbpedia"]:
        label_words = [
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
        label_words = [
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

    return [templates[idx] % label_word for label_word in label_words]
