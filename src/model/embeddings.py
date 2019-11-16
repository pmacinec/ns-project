import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_sequences_and_word_index_table(texts, max_words=None):
    """
    Get sequences and word index table from texts.

    :param texts: list, array of texts (strings).
    :param max_words: maximum number of words to preserver (top words).
    :return: (numpy.ndarray, dict), generated sequences and word index.
    """
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    
    # Create sequences from texts
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences
    sequences = pad_sequences(sequences, padding='post')

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
    Function to get embeddings matrix from word index and pre-trained_
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
