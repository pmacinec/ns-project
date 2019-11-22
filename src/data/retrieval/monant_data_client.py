import requests
import json
import time
from os.path import exists
from os import mkdir


class CentralStorageClient:
    """
    Client to perform authorized requests on Monant platform.

    :param token: str (static), token to be used in authorization.
    :param username: str, username for authorization to central storage.
    :param password: str, password for authorization to central storage.
    :param api_host: str, url of Monant platform API.
    :param data_folder: str, path of folder where data will be stored.
    """
    token = None

    def __init__(self, username, password, api_host, data_folder):
        """Create object of the class CentralStorageClient."""
        self.username = username
        self.password = password
        self.api_host = api_host
        self.data_folder = data_folder

        if not exists(f'{self.data_folder}/articles'):
            mkdir(f'{self.data_folder}/articles')

        if not exists(f'{self.data_folder}/annotations'):
            mkdir(f'{self.data_folder}/annotations')

        if not CentralStorageClient.is_authorized():
            self.authorize()

    @staticmethod
    def is_authorized():
        """
        Check whether client is authorized to Monant platform API.

        :return: bool, check result.
        """
        return CentralStorageClient.token is not None

    def authorize(self):
        """
        Authorize client to Monant platform.

        Authorize client with POST request (/auth} on Monant platform.

        :return: bool, result of authorization request.
        """
        login_data = {
            'username': self.username,
            'password': self.password,
        }
        r = requests.post(f'{self.api_host}/auth', json=login_data)

        if r.status_code == 200:
            CentralStorageClient.token = r.json()['access_token']

            return True

        return False

    def get_authorization_token(self):
        """
        Get authorization token.

        Get token to be authorized to Monant platform API. If client
        has been already authorized, token will be returned, otherwise
        authorization is performed at first.

        :return: str, authorization token.
        """
        if not CentralStorageClient.is_authorized():
            self.authorize()

        return CentralStorageClient.token

    def get_request_headers(self):
        """
        Get headers list to handle authorization in requests.

        :return: dict, request headers.
        """
        return {
            'Authorization': 'JWT ' + self.get_authorization_token()
        }

    def get_data(self):
        """
        Get articles data from Monant platform and save to JSON files.
        """
        has_next_page = True
        page = 1
        while has_next_page:
            print(f'Getting page {page}')
            response = self.get_articles(
                    page=page,
                    size=200,
                    order_by='extracted_at',
                    order_type='asc'
            )
            pagination = response.get('pagination')
            has_next_page = pagination.get('has_next')
            self.save_articles(response.get('articles'))
            page += 1
            time.sleep(2.5)

    def save_articles(self, articles):
        """
        Save articles into JSON files.

        :param articles: dict, articles objects.
        """
        for article in articles:
            with open(
                    f'{self.data_folder}/articles/{article.get("id")}.json',
                    'w'
            ) as f:
                json.dump(article, f)

    def get_articles(
            self,
            size=None,
            page=None,
            order_type='asc',
            order_by=None
    ):
        """
        Get articles from Monant platform.

        :param size: int (default: None), size of the result - number
            of articles to be returned.
        :param page: int (default: None), page to be shown.
        :param order_type: str (default: None), way of ordering results
            (asc or desc).
        :param order_by: str (default: None), column used for ordering.
        :return: dict , list of articles and pagination.
        """
        params = {}

        if order_type and order_by:
            params['order_type'] = order_type
            params['order_by'] = order_by
        if size:
            params['size'] = size
        if page:
            params['page'] = page

        params_str = "&".join(f'{k}={v}' for k, v in params.items())

        r = requests.get(
            f'{self.api_host}/v1/articles',
            params=params_str,
            headers=self.get_request_headers()
        )

        response = None
        if r.status_code == 200:
            response = r.json()

        return response

    def save_annotations(self):
        """
        Get entity annotations from Monant platform and save them to
        JSON files.
        """
        r = requests.get(
            f'{self.api_host}/v1/entity-annotations?'
            f'annotation_type=Source reliability (binary)&size=100',
            headers=self.get_request_headers()
        )

        entity_annotations = r.json().get('entity_annotations')

        for annotation in entity_annotations:
            annotation_id = annotation.get('entity_id')
            with open(
                    f'{self.data_folder}/annotations/{annotation_id}.json',
                    'w'
            ) as f:
                json.dump(annotation, f)
