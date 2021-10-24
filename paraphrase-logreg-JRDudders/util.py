# TODO: lab, homework
def parse_sts(data_file):
    """
    Reads a tab-separated sts benchmark file and returns
    texts: list of tuples (text1, text2)
    labels: list of floats
    """
    texts = []
    labels = []

    with open(data_file, 'r', encoding='utf-8') as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(float(fields[4]))
            t1 = fields[5].lower()
            t2 = fields[6].lower()
            texts.append((t1, t2))
    return texts, labels

# TODO: lab, homework
def sts_to_pi(texts, sts_labels, max_nonparaphrase, min_paraphrase):
    """Convert a dataset from semantic textual similarity to paraphrase.
    Remove any examples that are > max_nonparaphrase and < min_nonparaphrase."""

    pi_rows = [i for i,label in enumerate(sts_labels)
               if label >=min_paraphrase or label<=max_nonparaphrase]
    pi_texts = [texts[i] for i in pi_rows]

    # using indexing to get the right rows out of labels
    pi_labels = sts_labels[pi_rows]
    # convert to binary using threshold
    pi_labels = pi_labels > max_nonparaphrase

    return pi_texts, pi_labels