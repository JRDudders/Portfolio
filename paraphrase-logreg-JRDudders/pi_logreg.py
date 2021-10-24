import argparse
from util import parse_sts, sts_to_pi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk import word_tokenize, edit_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

def load_X(sent_pairs, tfidf_vectorizer):
    """Create a matrix where every row is a pair of sentences and every column in a feature.
    Feature (column) order is not important to the algorithm."""

    features = ["Word Error Rate", "BLEU", "Tfidf"]

    X = np.zeros((len(sent_pairs), len(features)))

    for i, sent in enumerate(sent_pairs):
        #Thing 1, get word edit distances:
        text1, text2 = sent
        t1_tokens = word_tokenize(text1)
        t2_tokens = word_tokenize(text2)

        word_edit_dist = edit_distance(t1_tokens, t2_tokens)
        wornums = (word_edit_dist / len(t1_tokens)) + (word_edit_dist / len(t2_tokens))
        X[i, 0] = wornums

        #Thing 2, implement BLEU scores:
        smooth = SmoothingFunction()

        try:
            bleu_first = sentence_bleu([t1_tokens,], t2_tokens, smoothing_function=smooth.method0)
        except Exception as e:
            print(e)
            bleu_first = 0.0
        finally:
            pass

        try:
            bleu_second = sentence_bleu([t2_tokens,], t1_tokens, smoothing_function=smooth.method0)
        except Exception as e:
            print(e)
            bleu_second = 0.0
        finally:
            pass
        X[i, 1] = bleu_first + bleu_second

        #Thing 3, get TFIDF

        # Uses a threshold of 0.7 to convert each similarity score into a paraphrase prediction

        # each item is a 2-tuple
        # this means we will get a (2, |vocab|) sparse representation back
        pair_reprs = tfidf_vectorizer.transform(sent)
        pair_similarity = cosine_similarity(pair_reprs[0], pair_reprs[1])
        X[i, 2] = pair_similarity[0,0]

    return X


def main(sts_train_file, sts_dev_file):
    """Fits a logistic regression for paraphrase identification, using string similarity metrics as features.
    Prints accuracy on held-out data. Data is formatted as in the STS benchmark"""

    min_paraphrase = 4.0
    max_nonparaphrase = 3.0

    # TODO 1: Load data partitions and convert to paraphrase dataset as in the lab
    # You will train a logistic regression on the TRAIN parition
    train_texts_sts, train_y_sts = parse_sts(sts_train_file)
    labels = np.asarray(train_y_sts)
    pi_texts, pi_labels = sts_to_pi(train_texts_sts, labels, max_nonparaphrase, min_paraphrase)

    # You will evaluate predictions on the VALIDATION partition
    dev_texts_sts, dev_y_sts = parse_sts(sts_dev_file)
    vectorizer = TfidfVectorizer("content", lowercase=True, analyzer="word", use_idf=True, min_df=10)
    all_t1, all_t2 = zip(*pi_texts)
    all_texts = all_t1 + all_t2
    vectorizer.fit(all_texts)
    cos_sims = []
    for pair in pi_texts:
        # each item is a 2-tuple
        # this means we will get a (2, |vocab|) sparse representation back
        pair_reprs = vectorizer.transform(pair)
        pair_similarity = cosine_similarity(pair_reprs[0], pair_reprs[1])
        cos_sims.append(pair_similarity[0, 0])

    X = load_X(pi_texts,vectorizer)

    dev_texts_sts, dev_y_sts = parse_sts(sts_dev_file)
    labels = np.asarray(dev_y_sts)
    dev_texts, dev_labels = sts_to_pi(dev_texts_sts, labels, max_nonparaphrase, min_paraphrase)
    dev_X = load_X(dev_texts, vectorizer)


    # TODO 2: Train a logistic regression model using sklearn.linear_model.LogisticRegression
    # Hint: The interface is very similar to other sklearn models we have used in class
    lr = LogisticRegression()
    lr.fit(X, pi_labels)

    # TODO 3: Evaluate your logistic regression model using accuracy, precision, recall and F1
    # Get predictions for the dev partition to do this
    dev_predict = lr.predict(dev_X)
    accuracy_something = accuracy_score(dev_labels, dev_predict)
    print(f'LR model accuracy score:{accuracy_something}')

    f1 = f1_score(dev_labels, dev_predict)
    print(f'LR model F1 Score:{f1}')

    precision = precision_score(dev_labels, dev_predict)
    print(f'LR Precision: {precision}')

    recall = recall_score(dev_labels, dev_predict)
    print(f'LR recall: {recall}')

    # TODO 4: You will need the same evaluation metrics for the baseline in the lab on the dev partition.
    # You may choose to do it in this script or to add it to the end of your lab script
    # Note: the results will be different because of differences in vocabulary collection; you may report either
    p = precision_score(pi_labels, dev_predict)
    r = recall_score(pi_labels, dev_predict)
    f = f1_score(pi_labels, dev_predict)
    a = accuracy_score(pi_labels, dev_predict)

    print(f"Baseline Scores: precision {p:0.03}\trecall {r:0.03}\tf1 {f:0.03}\taccuracy {a:0.03}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_dev_file", type=str, default="../strings-for-similarity-JRDudders/stsbenchmark/sts-dev.csv",
                        help="dev file")
    parser.add_argument("--sts_train_file", type=str, default="../strings-for-similarity-JRDudders/stsbenchmark/sts-train.csv",
                        help="train file")
    args = parser.parse_args()

    main(args.sts_train_file, args.sts_dev_file)
