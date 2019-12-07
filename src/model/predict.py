from argparse import ArgumentParser
import pandas as pd
import sys
from os.path import dirname, join, abspath
from tensorflow import keras
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append(abspath(join(dirname(__file__), '../../')))
from src.data.preprocessing import preprocess_data


def load_model(training_name):
    """
    Load model from specific training.

    :param training_name: str, name of training (also name of model
        stored in models folder.
    :return keras.Model, pre-trained keras model (our fake news
        detection net model).
    """
    if training_name is None:
        return None

    path_to_model = join(
        dirname(__file__),
        f'../../models/{training_name}/model'
    )

    return keras.models.load_model(path_to_model)


def parse_arguments():
    """
    Parse script call arguments.

    :return: argparse.Namespace, arguments namespace object.
    """
    parser = ArgumentParser()

    parser.add_argument("-f", "--file", dest="file", required=True,
                        help="File with text of article.")
    parser.add_argument("-m", "--model", dest="model_name", required=True,
                        help="Model name (folder, where model is stored ).")
    return parser.parse_args()


def get_text_dataframe(file):
    """
    From given text, return dataframe with one row only.

    :param file: str, path to file with text.
    :return pd.DataFrame, dataframe from given text.
    """
    with open(file, 'r') as f:
        text = f.read()
    return pd.DataFrame(
        data={'body': [text], 'label': ['not_predicted_yet']}
    )


def preprocess_input(dataframe, training_name):
    data = preprocess_data([dataframe])[0]
    print(data)
    word_index = pickle.load(
        open(
            join(
                dirname(__file__),
                f'../../models/{training_name}/word_index.obj'
            ),
            'rb'
        )
    )
    sequence = [word_index.get(word, 0) for word in data.loc[0].body.split()]

    return pad_sequences([sequence], padding='post')


def predict(model_name=None, file=None):
    print(model_name)
    model = load_model(model_name)

    dataframe = get_text_dataframe(file)

    to_predict = preprocess_input(dataframe, model_name)
    print(to_predict)
    prediction = model.predict(to_predict)
    print(prediction)
    # TODO add result
    print('According to model, the text you entered is ...')


if __name__ == "__main__":

    args = parse_arguments()

    if args.model_name is None:
        sys.exit('Argument --model/-m is required. Cannot predict without '
                 'model being loaded.')

    if args.file is None:
        sys.exit('Argument --file/-f is required. Please, set text to be '
                 'predicted.')

    predict(
        model_name=args.model_name,
        file=args.file
    )
