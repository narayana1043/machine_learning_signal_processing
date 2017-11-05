import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter


lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


def create_lexicon(pos, neg):

    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as file:
            contents = file.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)

    # print(l2.__len__())
    return l2


def sample_handling(sample, lexicon, classification):

    feature_set = []
    with open(sample, 'r') as file:
        contents = file.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            feature_set.append([features, classification])

    return feature_set


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1, 0])
    features += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size*len(features))

    x_train = list(features[:, 0][:-testing_size])
    y_train = list(features[:, 1][:-testing_size])

    x_test = list(features[:, 0][testing_size:])
    y_test = list(features[:, 1][testing_size:])

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = create_feature_sets_and_labels(
        pos='./pos.txt',
        neg='./neg.txt',
        test_size=0.1)

    with open('./sentiment_nn/sentiment_set.pickle', 'wb') as f:
        pickle.dump([x_train, y_train, x_test, y_test], f)




