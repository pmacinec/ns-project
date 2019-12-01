from argparse import ArgumentParser
import pandas as pd
import sys
from os.path import dirname, join, exists, isfile
from tensorflow import keras

# TODO add docstrings


def load_model(training_name):
    if training_name is None:
        return None

    path_to_model = join(
        dirname(__file__),
        f'../../models/{training_name}/model.ckpt'
    )
    if not exists(path_to_model):
        return None

    return keras.models.load_model(path_to_model)


def parse_arguments():
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", dest="data_path",
                       help="Path to data csv file.")
    group.add_argument("-t", "--text", dest="text",
                       help="Text to be predicted (e.g. news article).")
    parser.add_argument("-m", "--model", dest="model_name",
                        help="Model name (folder, where model is stored ).")
    return parser.parse_args()


def get_data(data_path=None, text=None):
    if text is not None:
        return pd.DataFrame(
            data={'body': [text], 'label': ['not_predicted_yet']}
        )

    if data_path is not None and isfile(data_path):
        return pd.read_csv(data_path)

    return None


def predict(model_name=None, data_path=None, text=None):

    model = load_model(model_name)

    data = get_data(data_path, text)
    if data is None:
        sys.exit('Sorry, there was an error reading your data.')

    # TODO make preprocessing and turn into sequences
    to_predict = ...

    predictions = model.predict(to_predict)
    if text is not None:
        # TODO add result
        print('According to model, the text you entered is ...')
    else:
        # TODO save predictions
        # TODO add path
        path = ...
        print(f'Predictions are stored in csv file in path: {path}.')


if __name__ == "__main__":

    args = parse_arguments()

    if args.model_name is None:
        sys.exit('Argument --model/-m is required. Cannot predict without '
                 'model being loaded.')

    predict(
        model_name=args.model_name,
        data_path=args.data_path,
        text=args.text
    )
