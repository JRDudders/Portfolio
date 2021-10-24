from util import split_data, labels_to_key, parse_federalist_papers, labels_to_y, find_zero_rule_class, \
    apply_zero_rule
from sklearn.metrics import accuracy_score
import argparse

def main(data_file):
    print(data_file)

    # TODO 1: load the data by defining parse_federalist_papers in util
    authors, essays, essay_ids = parse_federalist_papers(data_file)
    num_essays = len(essays)
    print(f"1: Working with {num_essays} reviews")

    # TODO 2: create a key that links author id string -> integer by defining labels_to_key in util
    author_key = labels_to_key(authors)
    print(f"2: Author key {author_key}")

    # TODO 3: convert the list of strings in `authors` to a np array by defining labels_to_key in util
    # convert all the labels using the key
    y = labels_to_y(authors, author_key)
    assert y.size == len(authors), f"Size of label array (y.size) must equal number of labels {len(authors)}"

    # shuffle and split the data. Function is already defined.
    train, test = split_data(essays, y, 0.3, shuffle=True)
    print(f"{len(train[0])} in train; {len(test[0])} in test")

    # TODO 4: define find_zero_rule and apply_zero_rule in util
    # the code below learns the zero rule on train data and then uses it to classify held-out essays
    train_y = train[1]
    most_frequent_class = find_zero_rule_class(train_y)
    # lookup label string from class #
    reverse_author_key = {v:k for k,v in author_key.items()}
    print(f"2. The most frequent class is {reverse_author_key[most_frequent_class]}")

    # apply zero rule to test reviews
    test_predictions = apply_zero_rule(test[0], most_frequent_class)
    print(f"3. Zero rule predictions on held-out data: {test_predictions}")

    # Report the accuracy of zero-rule predictions
    test_y = test[1]
    test_accuracy = accuracy_score(test_y,test_predictions)
    print(f"4. Accuracy of zero rule: {test_accuracy:0.03f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test supervised learning utilities')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author dataset')
    args = parser.parse_args()

    main(args.path)
