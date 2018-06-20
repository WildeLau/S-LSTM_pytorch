import numpy as np
import collections
import pickle
import re
import os
import sys


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print('Made directory %s' % path)


def text_preprocessing(dataset):
    """
        dataset: a combination of train, dev, test data for a single task
    """
    dataset_text = []
    dataset_label = []
    for file in dataset:
        lines = []
        labels = []
        with open(file) as f:
            for line in f:
                line = re.split('\s|-', line.lower().strip())
                words = line[1:]
                label = int(line[0])
                lines += [words]
                labels += [label]
        dataset_text += [lines]
        dataset_label += [labels]
    return dataset_text, dataset_label


def make_words_list(dataset):
    all_words = []
    for lines in dataset:
        for line in lines:
            all_words += line
    counter = collections.Counter(all_words)
    return counter


def word_to_idx(dataset_text, dataset_label, word_id_dict):
    transformed_text = []
    transformed_label = []
    for lines, labels in zip(dataset_text, dataset_label):
        new_lines = []
        new_label = []
        for line, label in zip(lines, labels):
            words = [word_id_dict[w] if w in word_id_dict else 1 for w in line]
            new_lines += [words]
            new_label += [label]
        transformed_text += [new_lines]
        transformed_label += [new_label]
    return transformed_text, transformed_label


def load_glove(glove_filename, dimensions=None):
    if dimensions is None:
        raise RuntimeError("The dimensions of glove need to be spcified.")
    glove_dict = dict()
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split()
            word = ' '.join(line[0: len(line)-300])
            embedding = [float(x) for x in line[len(line)-300:]]
            glove_dict[word] = embedding
    return glove_dict


def select_embedding_matrix(embedding_dict, word_dict,
                            start_offset=0, demensions=300):
    embedding_matrix = list()
    if start_offset > 0:
        for _ in range(start_offset):
            embedding_matrix.append(
                np.random.normal(0, 0.1, demensions).tolist())

    missing = 0
    for idx, word in sorted(zip(word_dict.values(), word_dict.keys())):
        try:
            embedding_matrix.append(embedding_dict[word])
        except KeyError:
            embedding_matrix.append(
                np.random.normal(0, 0.1, demensions).tolist())
            missing += 1
    pickle.dump(embedding_matrix,
                open('embedding/'+'apparel'+'_embedding_matrix', 'wb'))
    print("Total %d words are not in embedding dict." % missing)
    print(np.array(embedding_matrix).shape)


if __name__ == '__main__':
    # for test
    np.random.seed(1)
    make_dir('parsed_data')

    dataset = ['mtl_data/'+'apparel'+'.task.train',
               'mtl_data/'+'apparel'+'.task.test',
               'mtl_data/'+'apparel'+'.task.test']
    dataset_text, dataset_label = text_preprocessing(dataset)

    all_words = make_words_list(dataset_text)
    # Follow the offical tensorflow implemention. I don't know why
    vocab_size = len(all_words) - 2
    common_words = dict(all_words.most_common(vocab_size))
    print("Select %d words as common words." % len(common_words))

    counter = 2
    for key in common_words:
        common_words[key] = counter
        counter += 1

    transformed_text, transformed_label = word_to_idx(dataset_text,
                                                      dataset_label,
                                                      common_words)
    pickle.dump((transformed_text, transformed_label),
                open('parsed_data/'+'apparel'+'_dataset', 'wb'))

    glove_filename = 'embedding/glove.6B.300d.txt'
    glove_dict = load_glove(glove_filename, dimensions=300)

    select_embedding_matrix(glove_dict, common_words, start_offset=2)
