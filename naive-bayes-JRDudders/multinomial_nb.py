#!/usr/bin/env python
import argparse
from util import load_function_words, parse_federalist_papers, labels_to_key, labels_to_y, split_data, apply_zero_rule, find_zero_rule_class
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

def load_features(list_of_essays, list_of_features):
    total_essays = len(list_of_essays)
    total_features = len(list_of_features)
    essay_features = np.zeros((total_essays, total_features), dtype=np.int64)
    for i, essay in enumerate(list_of_essays):
        review_words_count = Counter(essay.lower().split())
        for j, word in enumerate(list_of_features):
            if word in review_words_count:
                essay_features[i, j] = review_words_count[word]
    return essay_features


def main(data_file, vocab_path):
    """Build and evaluate Naive Bayes classifiers for the federalist papers"""

    function_words = load_function_words(vocab_path)
    authors, essays, essay_ids = parse_federalist_papers(data_file)


    # TODO 1. define the function above to load attributed essays into a feature matrix
    X = load_features(essays, function_words)
    print(f"Numpy feature array has shape {X.shape} and dtype {X.dtype}")

    # TODO 2: load the author names into a vector y, mapped to 0 and 1, using functions from util
    labels_map = labels_to_key(authors)
    y = np.asarray(labels_to_y(authors, labels_map))


    # TODO 3: shuffle, then split the attributed data using util. Assign 75% train / 25% test
    # (TODO 3-6 will be evaluated by checking object/function use in code and via writeup in README
    # (..., I suggest using print statements to get info for the writeup)
    train, test = split_data(X, y, .25)
    train_X, train_y = train
    test_X, test_y = test

    print(f"Training data features shape:{train_X.shape}")

    # TODO 4: train a multinomial NB model, evaluate on validation split
    NB_multi = MultinomialNB()
    NB_multi.fit(train_X, train_y)
    print(f"Multinomial score: {NB_multi.score(test_X, test_y)}")

    # TODO 5: train a Bernoulli NB model, evaluate on validation split
    NB_bernie = BernoulliNB()
    binary_train_X = train_X > 0
    binary_test_X = test_X > 0
    NB_bernie.fit(binary_train_X, train_y)
    print(f"Bernoulli score: {NB_bernie.score(binary_test_X, test_y)}")

    # TODO 6: fit the zero rule, evaluate on validation split
    zero = find_zero_rule_class(train_y)
    find_zero = apply_zero_rule(test_X, zero)
    accuracy = np.mean(test_y == find_zero)
    print(f'Accuracy after zero rule: {accuracy}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes homework')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author dataset')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.function_words_path)
