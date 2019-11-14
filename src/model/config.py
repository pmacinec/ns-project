import json
from os.path import dirname, isfile
from argparse import ArgumentParser


# TODO add script call params into README, also do not forget to mention
# TODO that config in file is overwritten by args passed by script call

# TODO add steps how to create config file into README


# TODO add docstrings
def load_config(path=None):
    config_path = f'{dirname(__file__)}/config.json' if path is None else path

    if not isfile(config_path):
        print(f'Config file {path} not found.')
        return None

    with open(config_path, 'r') as config:
        return json.load(config)


# TODO add docstrings
def parse_input_parameters():
    parser = ArgumentParser()

    parser.add_argument("-f", "--file", dest="config_file",
                        help="Path to config file.")
    parser.add_argument("-bs", "--batch_size", dest="batch_size",
                        help="Batch size to be used in training.")
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate",
                        help="Learning rate to be used in training.")
    parser.add_argument("-hl", "--num_hidden_layers", dest="num_hidden_layers",
                        help="Number of hidden layers.")
    parser.add_argument("-l", "--logs", dest="logs_folder",
                        help="Path to logs folder")

    return parser.parse_args()


# TODO add docstrings
def load_custom_configs(config, args):
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate

    if args.num_hidden_layers is not None:
        config['num_hidden_layers'] = args.num_hidden_layers

    if args.logs_folder is not None:
        config['logs_folder'] = args.logs_folder

    return config


# TODO add docstrings
def get_config(args):
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
