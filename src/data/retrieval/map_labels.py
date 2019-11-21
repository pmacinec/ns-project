import json
import os


def get_annotations(data_folder):
    """
    Return annotations dictionary from json files.

    :param data_folder: str, folder where data are stored.
    :return: dict, annotations dictionary.
    """
    annotations_files = os.listdir('data_annotations')

    annotations = {}
    for file_name in annotations_files:
        annotation = json.load(
            open(f'{data_folder}/annotations/{file_name}', 'r')
        )
        key = int(annotation['entity_id'])
        annotations[key] = annotation['value']['value']

    return annotations


def annotate_articles(data_folder):
    """
    Annotate articles using annotations json files.

    :param data_folder: str, folder where data are stored.
    """
    articles_files = os.listdir('data')
    annotations = get_annotations()
    
    all_articles = {}
    for index, file_name in enumerate(articles_files):
        article = json.load(open(f'{data_folder}/articles/{file_name}', 'r'))
        print(f'{index} - {article["source"]["id"]}')
        article['label'] = annotations.get(article['source']['id'], None)
        all_articles[article['id']] = filter_data(article)

    with open(f'{data_folder}/dataset.json', 'w') as f:
        json.dump(all_articles, f)


def get_image(article):
    """
    Get image url from article dictionary.

    :return: str, url of image from article.
    """
    image_url = None
    media = article.get('media', None)
    if media is not None:
        for m in media:
            media_type = m['media_type'].get('name', None) 
            if media_type == 'image':
                image_url = m['url']
                break
    
    return image_url


def filter_data(article):
    """
    Filter only needed attributes from article dictionary.

    :param article: dict, article object.
    :return: dict, article object with filtered attributes.
    """
    filtered = {
        'id': article['id'],
        'title': article['title'],
        'perex': article['perex'],
        'body': article['body'],
        'author': article['author'].get('name', None) 
                    if article['author'] is not None 
                    else None,
        'image': get_image(article),
        'source': article['source']['name'],
        'label': article['label']
    }

    return filtered
