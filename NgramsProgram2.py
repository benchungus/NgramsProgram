import pickle
from nltk import word_tokenize
from nltk.util import ngrams


## use laplace smoothing to caluculate probability for a language
def compute_prob(text, unigram_dict, bigram_dict, V):
    unigrams_test = word_tokenize(text)
    bigrams_test = list(ngrams(unigrams_test, 2))

    p_lapalace = 1

    for bigram in bigrams_test:
        n = bigram_dict[bigram] if bigram in bigram_dict else 0
        d = unigram_dict[bigram[0]] if bigram[0] in unigram_dict else 0
        p_lapalace = p_lapalace * ((n+1)/(d+V))
    return p_lapalace

## unpickle all of our pickled files
with open('bigram_dict_English.pickle', 'rb') as f:
    english_bigram_dict = pickle.load(f)

with open('unigram_dict_English.pickle', 'rb') as f:
    english_unigram_dict = pickle.load(f)

with open('bigram_dict_French.pickle', 'rb') as f:
    french_bigram_dict = pickle.load(f)

with open('unigram_dict_French.pickle', 'rb') as f:
    french_unigram_dict = pickle.load(f)

with open('bigram_dict_Italian.pickle', 'rb') as f:
    italian_bigram_dict = pickle.load(f)

with open('unigram_dict_Italian.pickle', 'rb') as f:
    italian_unigram_dict = pickle.load(f)

## get the lines for test and solution files
with open('data/LangId.test', 'r', encoding='utf8') as f:
    test_lines = f.readlines()

with open('data/LangId.sol', 'r', encoding='utf8') as f:
    sol_lines = f.readlines()

## go through the test file lines and process them
correct = 0
incorrect_list = []

## write solutions to a file
with open('output.txt', 'w', encoding='utf8') as f:
    for i, line in enumerate(test_lines):
        sol_line = sol_lines[i].split(' ')[1].strip()

        ## get probability of all languages for the given line and then find which has most probability
        english_prob = compute_prob(line, english_unigram_dict, english_bigram_dict, len(english_unigram_dict))
        french_prob = compute_prob(line, french_unigram_dict, french_bigram_dict, len(french_unigram_dict))
        italian_prob = compute_prob(line, italian_unigram_dict, italian_bigram_dict, len(italian_unigram_dict))
        probs = [("English", english_prob), ("French", french_prob), ("Italian", italian_prob)]
        max_prob = max(probs, key=lambda x: x[1])
        format_prob = max_prob[0].strip()

        ## write max prob language to file
        f.write(format_prob + '\n')

        ## if matches with sol file, increase accuracy, if not, add it to incorrect list
        if format_prob == sol_line:
            correct += 1
        else:
            print(format_prob + " and " + sol_line)
            incorrect_list.append((max_prob[1], i+1))

## print the accuracy
accuracy = correct / len(test_lines)
print('Accuracy: {:%}'.format(accuracy))

## print wrong line info
print('Accuracy of wrong lines, and their line numbers: ', incorrect_list)