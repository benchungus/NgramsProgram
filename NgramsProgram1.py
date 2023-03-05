from nltk import word_tokenize
from nltk.util import ngrams
import pickle

## tokenizes and processes a file into the unigram and bigram dicts
def process_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().replace('\n', ' ')
    
    ## tokenize the text and form bigrams + unigrams
    unigrams = word_tokenize(text)
    bigrams = list(ngrams(unigrams, 2))
    
    ## generate bigram dict
    bigram_dict = {b:bigrams.count(b) for b in set(bigrams)}
    
    ## generate unigram dict
    unigram_dict = {t:unigrams.count(t) for t in set(unigrams)}
    
    ## return dictionaries
    return unigram_dict, bigram_dict
    
## iterate through the languages and generate and pickle all dicts
filelist = ["English", "French", "Italian"]
for i in filelist:
    filename = f'data/LangId.train.{i}'
    unigram_dict, bigram_dict = process_file(filename)
    with open(f'unigram_dict_{i}.pickle', 'wb') as f:
        pickle.dump(unigram_dict, f)
    with open(f'bigram_dict_{i}.pickle', 'wb') as f:
        pickle.dump(bigram_dict, f)