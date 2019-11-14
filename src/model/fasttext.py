import numpy as np
import os


def read_fastext_model(path):
    """
    Read fastText model from .vec file.

    :param path: str, path to model in .vec format.
    :return: dict, fastText pre-trained word embeddings.
    """
    if not os.path.isfile(path):
        return None

    fasttext = {}
    with open(path) as file:
        for line in file.readlines():
            values = line.split()
            fasttext[values[0]] = np.asarray(values[1:], dtype='float32')

    return fasttext
