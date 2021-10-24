import numpy as np
from collections import Counter
import json

def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words

# TODO: lab 1
def parse_federalist_papers(data_file):
    authors = []
    texts = []
    essay_ids = []
    with open(data_file, 'r') as df:
        data = json.load(df)
    for item in data:
        author, essay, text = item
        authors.append(author)
        texts.append(text)
        essay_ids.append(essay)
    return authors, texts, essay_ids

# TODO: write this function (lab1, homework)
def labels_to_key(labels):
    """
    Creates a mapping from string representations of labels to integers
    :param labels:
    :return: label_key, dict {str: int}
    """
    label_set = set(labels)
    label_key = {}
    for i, label in enumerate(label_set):
        label_key[label] = i
    return label_key

# TODO: write this function (lab1, homework)
def labels_to_y(labels, label_key):
    """
    :param labels: list of strings
    :param label_key: dictionary {str: int}
    :return: numpy vector y
    """
    y = np.zeros(len(labels), dtype=np.int)
    for i, l in enumerate(labels):
        y[i] = label_key[l]
    return y

# TODO: write this function (lab1, homework)
def find_zero_rule_class(train_y):
    """
    Determines the class predicted by the zero rule algorithm
    :param train_y: training labels
    :return: most_freq, the most frequent element in train_y
    """
    class_counts = Counter(train_y)
    print(class_counts)
    most_freq = max(class_counts, key=lambda k: class_counts[k])
    return most_freq

# TODO: write this function (lab1, homework)
def apply_zero_rule(X, zero_class):
    """
    Predicts most frequent class using zero rule algorithm
    :param X: iterable, data to classify
    :param zero_class: class to predict
    :return: classifications: numpy array
    """
    classifications = np.zeros(len(X), dtype=np.int)
    # assign every y the zero class
    classifications[:] = zero_class
    return classifications

# data split and shuffle functions
def shuffle_dataset(data0, data1):
    """
    Shuffles two iterables containing associated data in unison, e.g. X and y; X and file id's
    :param data0: iterable, e.g. X
    :param data1: iterable, e.g. y
    :return: tuple (shuffled0, shuffled1)
    """
    # seed random for consistency in student homework
    np.random.seed(521)
    # define a new order for the indices of data0
    new_order = np.random.permutation(len(data0))

    # cast inputs to np array
    data0 = np.asarray(data0)
    data1 = np.asarray(data1)

    # reorder
    shuffled0 = data0[new_order]
    shuffled1 = data1[new_order]
    return (shuffled0, shuffled1)

def split_data(data0, data1, test_percent = 0.3, shuffle=True):
    """
    Splits dataset for supervised learning and evaluation
    :param data0: iterable, e.g. X, features
    :param data1: iterable, e.g. y, labels corresponding to the features in X
    :param test_percent: percent data to assign to test set
    :param shuffle: shuffle data order before splitting
    :return: two tuples, (data0_train, data1_train), (data0_test, data1_test)
    """
    if shuffle:
        data0, data1 = shuffle_dataset(data0, data1)
    data_size = len(data0)
    num_test = int(test_percent * data_size)

    train = (data0[:-num_test], data1[:-num_test])
    test = (data0[-num_test:], data1[-num_test:])
    return train, test
