Paraphrase Identification using string similarity
---------------------------------------------------

This project examines string similarity metrics for paraphrase identification (PI).
It converts semantic textual similarity data to paraphrase identification data using threshholds.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting paraphrase.


Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

## Results

TODO: Write up your results for the baseline implemented in the lab and the logistic regression model in the homework:
Describe the dataset (1 sentence),
describe the baseline and the logistic regression (2+ sentences each), 
fill the table with evaluation on the dev partition,
and compare the results (3 sentences).

Precise results will vary a little based on preprocessing choices.


| Model Name | Accuracy | Precision | Recall | F1|
| ---------- | -------- | --------- | ------- | ---|
| (Fill me) | ...




## Homework: pi_logreg.py

* Train a logistic regression for paraphrase identification on the training data using three features:
    - BLEU
    - Word Error Rate
    - Cosine Similarity of TFIDF vectors
* Use the logistic regression implementation in `sklearn`.
* Update the readme as described in *Results*.

`python pi_logreg.py --sts_dev_file stsbenchmark/sts-dev.csv --sts_train_file stsbenchmark/sts-train.csv`

## lab.py

`lab.py` converts a STS dataset to PI and checks the distribution of paraphrase/nonparaphrase.
Then, it evaluates TFIDF vector similarity with a threshold of 0.7 as a model of paraphrase
using precision and recall.

Example usage:

`python lab.py --sts_data stsbenchmark/sts-dev.csv`


