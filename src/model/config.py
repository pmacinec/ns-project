import json
from os.path import dirname, isfile
from argparse import ArgumentParser


def load_config(path=None):
    """
    Load config from specified JSON file.

    :param path: str, path to config file (json format).
    :return: dict, config dictionary.
    """
    config_path = f'{dirname(__file__)}/config.json' if path is None else path

    if not isfile(config_path):
        print(f'Config file {path} not found.')
        return None

    with open(config_path, 'r') as config:
        return json.load(config)


def parse_input_parameters():
    """
    Parse script call arguments.

    :return: argparse.Namespace, arguments namespace object.
    """
    parser = ArgumentParser()

    parser.add_argument("-f", "--file", dest="config_file",
                        help="Path to config file.")
    parser.add_argument("-bs", "--batch-size", dest="batch_size",
                        help="Batch size to be used in training.")
    parser.add_argument("-lr", "--learning-rate", dest="learning_rate",
                        help="Learning rate to be used in training.")
    parser.add_argument("-hl", "--num-hidden-layers", dest="num_hidden_layers",
                        help="Number of hidden layers.")
    parser.add_argument("-e", "--epochs", dest="epochs",
                        help="Number of epochs to train.")
    parser.add_argument("-w", "--max-words", dest="max_words",
                        help="Maximum words in vocabulary to use.")
    parser.add_argument("-s", "--samples", dest="num_samples", default=None,
                        help="Number of samples from data.")
    parser.add_argument("-d", "--data", dest="data_file",
                        help="Path to data csv file.")
    parser.add_argument("-t", "--test-size", dest="test_size",
                        help="Train test split rate (test size).")
    parser.add_argument("-sl", "--max-sequence-len", dest="max_seq_len",
                        help="Maximum length of all sequences.")
    parser.add_argument("-lstm", "--lstm-units", dest="lstm_units",
                        help="Number of units in LSTM layer.")

    return parser.parse_args()


def load_custom_configs(config, args):
    """
    Overwrite default configs with script call arguments.

    :param config: dict, dictionary of config.
    :param args: argparse.Namespace, object of script call arguments.
    :return: dict, updated config.
    """
    args_names = ['batch_size', 'learning_rate', 'num_hidden_layers',
                  'epochs', 'max_words', 'num_samples', 'lstm_units',
                  'data_file', 'test_size', 'max_seq_len']

    for arg in args_names:
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)

    return config


def get_config(args):
    """
    Get config from file and script call arguments.

    :param args: argparse.Namespace, script call arguments object.
    :return: dict, config from file and script call arguments.
    """
    # Read default config file
    config = load_config()

    # Check config file from script call arguments
    if args.config_file is not None:
        config_external = load_config(args.config_file)

        if config_external is not None:
            config = config_external

    # Overwrite config with script call arguments
    config = load_custom_configs(config, args)

    return config
