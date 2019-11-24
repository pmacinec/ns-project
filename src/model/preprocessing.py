import numpy as np
import pandas as pd
from os.path import dirname, join
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def read_data(path=None, samples=None):
    """
    Read the data from specified csv file.

    :param path: str, path to data csv file.
    :param samples: int, number of samples to choose, None for all.
    :return: pd.DataFrame, dataframe with data.
    """
    if path is None:
        path = join(dirname(__file__), '../../data/preprocessed/dataset.csv')

    df = pd.read_csv(path, index_col=0)

    # Categorical encoding of labels
    df['label'] = df['label'].apply(
        lambda label: 1 if label == 'unreliable' else 0
    )

    if samples is not None:
        df = df.sample(int(samples))

    return df


def split_data(x, y, test_size=0.15):
    """
    Split data to train and test parts.

    :param x: list, list of train data.
    :param y: list, list of labels.
    :param test_size: float, rate of test size.
    :return: list, list of lists in format:
        x_train, x_test, y_train, y_test.
    """
    return train_test_split(x, y, test_size=test_size, random_state=1)


def get_sequences_and_word_index(texts, max_words=None, max_seq_len=None):
    """
    Get sequences and word index table from texts.

    :param texts: list, array of texts (strings).
    :param max_words: int, maximum number of words to preserve (top
        words).
    :param max_seq_len: int, maximum length of all sequences.
    :return: (numpy.ndarray, dict), generated sequences and word index.
    """
    if max_words is not None:
        max_words = int(max_words)

    if max_seq_len is not None:
        max_seq_len = int(max_seq_len)

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    
    # Create sequences from texts
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences
    sequences = pad_sequences(sequences, padding='post', maxlen=max_seq_len)

    word_index = tokenizer.word_index
    word_index['<pad>'] = 0
    if max_words is not None:
        # This step is done due to issue on keras Tokenizer:
        # https://github.com/keras-team/keras/issues/8092
        word_index = {
            word: idx for word, idx
            in tokenizer.word_index.items()
            if idx <= max_words
        }

    return sequences, word_index


def get_embeddings_matrix(word_index, pretrained_embeddings, embeddings_dim):
    """
    Function to get embeddings matrix from word index and pre-trained
    embeddings (e.g. fastText).

    :param word_index: dict, dictionary of format 'word: index'.
    :param pretrained_embeddings: np.array, pre-trained embeddings.
    :param embeddings_dim: int, embeddings dimension (len of vectors).
    :return: numpy.array, embeddings matrix.
    """
    # Words that does not exist in word index table,
    # will have vectors containing only zeros
    embeddings_matrix = np.zeros((len(word_index), embeddings_dim))

    not_found = 0
    for word, i in word_index.items():
        vector = pretrained_embeddings.get(word)
        if vector is not None:
            embeddings_matrix[i] = vector
        else:
            not_found += 1

    print(f'Number of words not found in pre-trained embeddings: {not_found}')
    return embeddings_matrix
