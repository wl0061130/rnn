#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random
import json

import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # Your Code here
    ##################
    data_length = len(vocabulary)
    with open('dictionary.json', encoding='utf-8') as inf:
        dictionary = json.load(inf, encoding='utf-8')

    raw_x = [dictionary.get(w, 0) for w in vocabulary]
    raw_y = [dictionary.get(w, 0) for w in vocabulary[1:]]
    raw_y.append(0)

    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)

    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i: batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i: batch_partition_length * (i + 1)]

    epoch_size = batch_partition_length // num_steps
    for i in range(epoch_size):
        x = data_x[:, num_steps * i: num_steps * (i + 1)]
        y = data_y[:, num_steps * i: num_steps * (i + 1)]
        yield (x, y)
