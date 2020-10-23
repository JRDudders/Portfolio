"""
Dudley_ngrams.py
Adapted from code by Paul Ebreo
Last changed 2020-10-20 by John-Rick Dudley
"""

from random import choice
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import argparse, sys


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", action="store", default="dickens.txt", help="Corpus to generate from")
parser.add_argument("-s", "--starter", action="store", help="Starting N-gram phrase") #my addition
options = parser.parse_args()
starter = options.starter



def get_counts(context_length, training_text):
    """
    This function counts the frequencies of all continuations of each context tuple
    :param context_length: Integer, number of tokens preceding current token (use 2 for trigrams)
    :param training_text: The training data as one big string
    :return: counts: A dictionary of context tuples to dictionaries of continuation counts
    """
    counts = {}

    tokens = word_tokenize(training_text)
    for i in range(len(tokens) - context_length):
        context = []
        next_token = tokens[i + context_length]

        for j in range(context_length):
            context.append(tokens[i + j])

        # Add 1 to frequency or create new dictionary item for this tuple
        if tuple(context) in counts:
            if next_token in counts[tuple(context)]:
                counts[tuple(context)][next_token] += 1
            else:
                counts[tuple(context)][next_token] = 1
        else:
            counts[tuple(context)] = {next_token: 1}

    return counts


def generate_from_file(context_length, training_file, output_length=60):
    global starter #my addition
    # Open the training file
    with open(training_file, 'r') as f:
        training_data = f.read()

    counts = get_counts(context_length, training_data)
    #My biggest contribution starts here:
    if starter is not None:
        starter = word_tokenize(starter)
        starter = tuple(starter)
    #A check to see if user input matches the context length specified by their n-gram generator
        if len(starter) < context_length:
            print("Your starting phrase was too short. Please retry with a phrase containing", context_length,
                  "tokens, OR adjust your context length.")
            sys.exit()
    #If user input is longer than expected, truncates it
        elif len(starter) > context_length:
            print("Your starting phrase was too long. Truncating the input to first", context_length,"tokens...")
            first_tokens = tuple(starter[:context_length])
        else:
            first_tokens = starter
    #Unspecified user input = random seleciton of tokens
    else:
        first_tokens = choice(list(counts.keys()))

    output_list = list(first_tokens)
    if first_tokens not in counts: #We were expected to check if the input was OOV, and if so proceed with random tokens
        if output_list[-1] == "." or output_list[-1] == "!" or output_list[-1] == "?":
            pass
        else:
            output_list.append(".")
        first_tokens = choice(list(counts.keys()))
    current_context = first_tokens

    for i in range(output_length):
        next_context = max(counts[current_context], key=(counts[current_context].get))
        temp = list(current_context)
        temp.pop(0)  # Remove first token in previous context
        temp.append(next_context)  # Add new token for the next context
        next_token = temp[-1]
        next_context = tuple(temp)

        current_context = next_context

        output_list.append(next_token)
    if output_list[-1] != "." and output_list[-1] != "!" and output_list[-1] != "?":
        output_list.append(".")
    TreebankWordDetokenizer().detokenize(output_list)
    print(" ".join(output_list))


generate_from_file(3,options.file)

#Found my Treebank line from this link: https://stackoverflow.com/questions/21948019/python-untokenize-a-sentence
