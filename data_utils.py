import os
import numpy as np
import pickle


def prepared_data(seqs, labels):
    lengths = [lens(s) for s in seqs]
    labels = np.array(labels).astype('int32')
    return [np.array(seqs), labels, numpy.array(lengths).astype('int32')]


def removed_unk(x, n_words):
    return [[1 if w >= n_words else w for w in sen] for sen in x]


def load_data(path, n_words):
    with open(path, 'rb') as f:
        dataset_x, dataset_y = pickle.load(f)
        train_set_x, train_set_y = dataset_x[0], dataset_y[0]
        dev_set_x, dev_set_y = dataset_x[1], dataset_y[1]
        test_set_x, test_set_y = dataset_x[2], dataset[2]

    train_set_x = removed_unk(train_set_x, n_words)
    dev_set_x = removed_unk(dev_set_x, n_words)
    test_set_x = removed_unk(test_set_x, n_words)
