import json
import sys
from os.path import dirname
from monant_data_client import CentralStorageClient
from map_labels import annotate_articles


def get_config():
    """
    Read configuration file for data retrieval.

    :return: dict, configuration.
    """
    return json.load(open(f'{dirname(__file__)}/config.json', 'r'))


def main():
    config = get_config()

    # Get the data from Monant platform
    client = CentralStorageClient(
        username=config['username'],
        password=config['password'],
        api_host=config['api_host'],
        data_folder=config['data_folder']
    )
    client.get_data()
    client.save_annotations()

    # Annotate articles
    annotate_articles(data_folder=config['data_folder'])


if __name__ == "__main__":
    sys.exit(main())
