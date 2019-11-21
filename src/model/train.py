import os
import gc
from os.path import dirname, join
import datetime
import numpy as np
from config import parse_input_parameters, get_config
from model import FakeNewsDetectionNet
from preprocessing import read_data, get_sequences_and_word_index, split_data,\
    get_embeddings_matrix
from fasttext import read_fasttext_model
import tensorflow.keras as keras


def get_model(dim_input, dim_embeddings, embeddings, optimizer):
    """
    Function to get compiled model, ready for training.

    :param dim_input: int, input dimension (vocabulary size).
    :param dim_embeddings: int, dimension of embeddings (e.g. 300 for
        pre-trained fastText embeddings).
    :param embeddings: np.ndarray, matrix of pre-trained embeddings.
    :param optimizer: str|keras.optimizers.Optimizer, optimizer to be
        used in training.
    :return: FakeNewsDetectionNet, compiled Keras model.
    """
    model = FakeNewsDetectionNet(
        dim_input=dim_input,
        dim_embeddings=dim_embeddings,
        embeddings=embeddings
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def prepare_data(
        data_path=None,
        max_words=None,
        test_size=0.15,
        max_seq_len=None,
        samples=None
):
    """
    Function to load and prepare data for training.

    :param data_path: str, path where csv file is stored.
    :param max_words: int, maximum number of top words in vocabulary.
    :param test_size: float, train test split rate.
    :param max_seq_len: int, maximum length of all sequences.
    :param samples: int, number of samples from data to choose.
    :return: list, list of data and word index in format:
        x_train, x_test, y_train, y_test, word_index
    """
    data = read_data(data_path, samples)

    labels = np.asarray(data['label'])
    sequences, word_index = get_sequences_and_word_index(
        data['body'], max_words, max_seq_len
    )

    print(f'Count of unique tokens: {len(word_index)}')
    print(f'Sequences shape: {sequences.shape}')

    x_train, x_test, y_train, y_test = split_data(sequences, labels, test_size)

    return x_train, x_test, y_train, y_test, word_index


def get_callbacks(logs_dir='logs', logs_name=None, checkpoint_path='models'):
    """
    Function to get callbacks for training.

    :param logs_dir: str, directory where tensor board logs are generated.
    :param logs_name: str, name of current logs.
    :param checkpoint_path: str, path where checkopints are stored. 
    :return: list, list of callbacks.
    """
    if logs_name is None:
        logs_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=join(dirname(__file__), f'../../{logs_dir}/{logs_name}'),
        histogram_freq=1,
        profile_batch=0
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=join(dirname(__file__), f'../../{checkpoint_path}/model.ckpt'),
        save_weights_only=True,
        verbose=1,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    return [tensorboard, checkpoint]


def train(config):
    """
    Function to train prepared model on chosen data.

    :param config: dict, configuration to be used.
    :return: FakeNewsDetectionNet, trained model.
    """
    # Read the data and get word index
    print('Preparing data...')
    x_train, x_test, y_train, y_test, word_index = prepare_data(
        data_path=config.get('data_file', None),
        max_words=config.get('max_words', None),
        test_size=config.get('test_size', None),
        max_seq_len=config.get('max_seq_len', None),
        samples=config.get('num_samples', None)
    )
    print(f'Data prepared. Vocabulary size: {len(word_index)}.')

    print('Reading fastText model...')
    # Read pre-trained fasttext embeddings
    fasttext = read_fasttext_model(
        join(dirname(__file__), '../../models/fasttext/wiki-news-300d-1M.vec')
    )

    print('Creating embeddings matrix...')
    # Get filtered embeddings matrix
    embeddings_matrix = get_embeddings_matrix(
        word_index,
        fasttext,
        300
    )

    vocabulary_size = len(word_index)

    del fasttext
    del word_index
    gc.collect()

    optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])

    model = get_model(vocabulary_size, 300, embeddings_matrix, optimizer)

    print('Training the model...')
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=config['batch_size'],
        validation_data=(x_test, y_test),
        callbacks=get_callbacks(),
        epochs=config['epochs']
    )

    model.summary()

    return model


if __name__ == "__main__":
    args = parse_input_parameters()

    config = get_config(args)
    print(f'Config for this training: {config}')

    train(config)
    print('Model training ended.')
