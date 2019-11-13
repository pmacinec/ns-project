from langdetect import detect
import re
import time
from sklearn.base import TransformerMixin
from nltk.tokenize import sent_tokenize


class ColumnsFilter(TransformerMixin):
    """
    Transformer to drop columns of the dataframe.

    :param columns: list, list of columns to drop.
    :param except_columns: list, list of columns to preserve.
    """

    def __init__(self, columns=None, all_except=None):
        self.columns = columns
        self.all_except = all_except

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):

        print('ColumnsFilter transformation started.')
        start_time = time.time()

        if self.columns is not None:
            columns = self.columns
        elif self.all_except is not None and len(self.all_except):
            columns = df[df.columns.difference(self.all_except)]
        else:
            return df

        df = df.drop(columns, axis=1)

        end_time = time.time()
        print(f'ColumnsFilter transformation ended, '
              f'took {end_time - start_time} seconds.')

        return df


class EmptyValuesFilter(TransformerMixin):
    """
    Filter empty values in dataset of selected columns.

    Empty values can be NaN or empty strings.

    :param columns: list, subset of columns to drop samples with empty
        values in.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        print('NanFilter transformation started.')
        start_time = time.time()

        df = df.dropna(subset=self.columns)

        for column in self.columns:
            df = df[df[column] != '']

        end_time = time.time()
        print(f'NanFilter transformation ended, took '
              f'{end_time - start_time} seconds.')

        return df


class ArticlesLanguageFilter(TransformerMixin):
    """
    Filter to remove articles written in different language.

    :param column: str, name of column to check the language in.
    :param language: str, shortcut of language (e.g. 'en').
    """

    def __init__(self, column, language):
        self.column = column
        self.language = language

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        print('ArticlesLanguageFilter transformation started.')
        start_time = time.time()

        df_copy = df.copy()

        df_copy['lang'] = df_copy[self.column].apply(
            lambda text: detect(text)
        )
        df_copy = df_copy[df_copy.lang == self.language]

        df_copy.drop(['lang'], axis=1, inplace=True)

        end_time = time.time()
        print(f'ArticlesLanguageFilter transformation ended, took '
              f'{end_time - start_time} seconds.')

        return df_copy


class ArticlesSizeFilter(TransformerMixin):
    """
    Filter to remove all articles that are too short or too long.

    Both attributes, upper and lower boundaries are not included
    in interval.

    :param column: str, name of column to check the size of article.
    :param lower_boundary: int, minimum words in article.
    :param upper_boundary: int, maximum words in article.
    """

    def __init__(self, column, lower_boundary, upper_boundary):
        self.column = column
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        print('ArticlesSizeFilter transformation started.')
        start_time = time.time()

        df_copy = df.copy()

        df_copy['num_words'] = df_copy[self.column].apply(
            lambda text: len(text.split())
        )
        df_copy = df_copy[
            (df_copy['num_words'] > self.lower_boundary) &
            (df_copy['num_words'] < self.upper_boundary)
            ]

        df_copy.drop(['num_words'], axis=1, inplace=True)

        end_time = time.time()
        print(f'ArticlesSizeFilter transformation ended, took '
              f'{end_time - start_time} seconds.')

        return df_copy


class ArticlesSentenceLengthFilter(TransformerMixin):
    """
    Filter all articles that have extreme values in sentences length.

    Both attributes, upper and lower boundaries are not included
    in interval.

    :param column: str, name of column to check the average sentence
        length of article.
    :param lower_boundary: int, minimum average sentence length.
    :param upper_boundary: int, maximum average sentence length.
    """

    def __init__(self, column, lower_boundary, upper_boundary):
        self.column = column
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary

    def fit(self, df, y=None, **fit_params):
        return self

    def get_avg_sentence_length(self, text, num_words):
        result = re.sub(r'\w\n', '. ', text)
        result = re.sub(r'\.{2,}', '. ', result)
        result = re.sub(r'\.', '. ', result)

        return num_words / len(sent_tokenize(result))

    def transform(self, df, **transform_params):
        print('ArticlesSentenceLengthFilter transformation started.')
        start_time = time.time()

        df_copy = df.copy()

        df_copy['num_words'] = df_copy[self.column].apply(
            lambda text: len(text.split())
        )
        df_copy['avg_sent_length'] = df_copy.apply(
            lambda sample: self.get_avg_sentence_length(
                sample[self.column],
                sample['num_words']
            ),
            axis=1
        )
        df_copy = df_copy[
            (df_copy['avg_sent_length'] > self.lower_boundary) &
            (df_copy['avg_sent_length'] < self.upper_boundary)
            ]

        df_copy.drop(['avg_sent_length', 'num_words'], axis=1, inplace=True)

        end_time = time.time()
        print(f'ArticlesSentenceLengthFilter transformation ended, took '
              f'{end_time - start_time} seconds.')

        return df_copy


class TextPreprocessor(TransformerMixin):
    """
    Transformer to clean text attribute.

    Following transformations are included:
    1. Making text lowercase.
    2. Removing html tags.
    3. Remove special characters.

    :param column: str, name of column to clean text in.
    """

    def __init__(self, column):
        self.column = column

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        print('TextPreprocessor transformation started.')
        start_time = time.time()

        df_copy = df.copy()

        # Lowercase text.
        df_copy[self.column] = df_copy[self.column].apply(
            lambda text: text.lower()
        )

        # Remove html characters.
        df_copy[self.column] = df_copy[self.column].apply(
            lambda text: re.sub(r'<.*?>', '', text)
        )
        # Remove other special characters.
        df_copy[self.column] = df_copy[self.column].apply(
            lambda text: re.sub(r'[^a-zA-Z0-9\.,?!]+', ' ', text)
        )
        # Remove urls
        df_copy[self.column] = df_copy[self.column].apply(
            lambda text: re.sub(r'(www|http:|https:)+[^\s]+[\w]', '', text)
        )
        # Remove xml specific strings
        df_copy[self.column] = df_copy[self.column].apply(
            lambda text: re.sub(r'<!--//<!\[CDATA\[[^\]]*\]\]>-->', '', text)
        )

        end_time = time.time()
        print(f'TextPreprocessor transformation ended, took '
              f'{end_time - start_time} seconds.')

        return df_copy


class DuplicatesFilter(TransformerMixin):
    """
    Filter to remove duplicate articles.

    :param column: str, name of column to check duplicates in.
    """

    def __init__(self, column):
        self.column = column

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        print('DuplicatesFilter transformation started.')
        start_time = time.time()

        df_copy = df.copy()

        df_copy.drop_duplicates(subset=[self.column], inplace=True)

        end_time = time.time()
        print(f'DuplicatesFilter transformation ended, took '
              f'{end_time - start_time} seconds.')

        return df_copy
