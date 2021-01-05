from collections import defaultdict

# Initialize dictionaries with simple Delta smoothing
start_c = defaultdict(int)
trans_c = defaultdict(lambda: defaultdict(int))
emit_c = defaultdict(lambda: defaultdict(int))

start_p = defaultdict(lambda: 0.00000001)
trans_p = defaultdict(lambda: defaultdict(lambda: 0.00000001))
emit_p = defaultdict(lambda: defaultdict(lambda: 0.00000001))

#tags list
allTags = []
prev_tag = "<s>"

# Task 1
# Split the training file en_gum-ud-train.conllu into a list of lines.
# For each line that contains a tab ("\t"),
# split it by tab to collect the word (second column) and part of speech tag (5th column)
with open("en_gum-ud-train.conllu", "r", encoding="utf-8") as f:
  train = f.readlines()

  for line in train:
      if "\t" in line:
          tok = line.strip().split("\t")[1]
          tag = line.strip().split("\t")[4]
          allTags.append(tag)

          if prev_tag == "<s>":
              start_c[tag] += 1               # start counts
          else:
              trans_c[prev_tag][tag] += 1     # transition counts

          emit_c[tag][tok] += 1               # emission counts
          prev_tag = tag

      else:
          prev_tag = "<s>"

# Tagger states Q (the tag set)
states = tuple(set(allTags))


# Viterbi algorithm to find best path
def viterbi(obs, states, start_p, trans_p, emit_p):
    path = [{}]
    backpath = []
    # Get initial probabilities for each tag given first token (obs[0])
    for tag in states:
        path[0][tag] = start_p[tag] * emit_p[tag][obs[0]]
    # Get subsequent probabilities for obs[t] where t > 0
    for tok_num in range(1, len(obs)):
        path.append({})
        backpointer = {}
        for tag in states:
            max_prob = 0.0
            probs = []
            for prev_tag in states:
                prob = path[tok_num - 1][prev_tag] * trans_p[prev_tag][tag] * emit_p[tag][obs[tok_num]]
                probs.append(prob)
                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev_tag
            path[tok_num][tag] = max_prob
            backpointer[tag] = best_prev
        backpath.append(backpointer)
    optimal_list = []

    # Get most probable tag at end of path
    current_best_tag = max(path[-1], key=path[-1].get)
    optimal_list.append(current_best_tag)
    backpath.reverse()
    for backpointer in backpath:
        optimal_list.append(backpointer[current_best_tag])
        current_best_tag = backpointer[current_best_tag]
    optimal_list.reverse()

    # The highest probability
    max_total_prob = max(path[-1].values())
    #print('Best sequence: ' + ' '.join(optimal_list) + ' with highest probability of ' + str(float(max_total_prob)))
    return max_total_prob


# Read test file, get observed tokens
text_test = open("en_gum-ud-dev.conllu", 'r', encoding="utf8").readlines()

test_words = []
test_tags = []
observed_toks = []
observed_tags = []
prev_tag = ""

for line in text_test:
    if '\t' in line:
        line = line.split('\t')
        tag = line[4]
        if tag not in allTags:
            allTags.append(tag)
        word = line[1]
        test_words.append(word)
        test_tags.append(tag)

        if line[0] == "1":
            if tag not in start_c.keys():
                start_c[tag] = 0
            start_c[tag] += 1
            prev_tag = ""
            if test_words is not None:
                observed_toks.append(tuple(test_words))
                observed_tags.append(tuple(test_tags))
                test_words.clear()
                test_tags.clear()

        if prev_tag is not None:
            trans_c[prev_tag][tag] += 1

        if tag not in emit_c.keys():
            emit_c[tag][word] = 0
        emit_c[tag][word] += 1
        prev_tag = tag
    else:
        pass

# Add new keys to dictionaries
for k, v in start_c.items():
    total = sum(start_c.values())
    percent = float(v/total)
    start_p[k] = percent

for tag in trans_c:
    total = sum(trans_c[tag].values())  # sum values for this pos
    for tok in trans_c[tag]:
        freq = trans_c[tag][tok]
        trans_p[tag][tok] = freq/total

for tag in emit_c:
    total = sum(emit_c[tag].values())  # sum values for this pos
    for tok in emit_c[tag]:
        freq = emit_c[tag][tok]
        emit_p[tag][tok] = freq/total


for i in observed_toks:
    viterbi(i, states, start_p, trans_p, emit_p)